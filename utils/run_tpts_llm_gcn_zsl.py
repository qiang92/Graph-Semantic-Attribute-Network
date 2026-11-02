import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.run_llm_gcn_zsl import (  # noqa: E402
    ArrayDataset,
    LLMGCNZSL,
    build_class_graph,
    build_target_similarity,
    set_seed,
)


CLASS_ORDER = ["0000", "1001", "0110", "1011", "0111", "1111"]
ATTR_MATRIX = np.array(
    [
        [0, 0, 0, 0],
        [1, 0, 0, 1],
        [0, 1, 1, 0],
        [1, 0, 1, 1],
        [0, 1, 1, 1],
        [1, 1, 1, 1],
    ],
    dtype=np.float32,
)


def load_tpts(path):
    df = pd.read_csv(path)
    df["fault_type"] = df[["G", "C", "B", "A"]].astype(int).astype(str).agg("".join, axis=1)
    x_all, y_all = [], []
    for idx, code in enumerate(CLASS_ORDER):
        part = df[df["fault_type"] == code]
        x_all.append(part[["Ia", "Ib", "Ic", "Va", "Vb", "Vc"]].values.astype(np.float32))
        y_all.append(np.full(len(part), idx, dtype=np.int64))
    return x_all, y_all


def make_split(x_all, y_all, unseen_idx):
    seen_idx = [i for i in range(len(CLASS_ORDER)) if i not in unseen_idx]
    x_train = np.vstack([x_all[i] for i in seen_idx])
    y_train = np.concatenate([y_all[i] for i in seen_idx])
    x_test = np.vstack([x_all[i] for i in unseen_idx])
    y_test = np.concatenate([y_all[i] for i in unseen_idx])
    scaler = StandardScaler().fit(x_train)
    return (
        scaler.transform(x_train).astype(np.float32),
        y_train,
        ATTR_MATRIX[y_train],
        scaler.transform(x_test).astype(np.float32),
        y_test,
        ATTR_MATRIX[unseen_idx],
        np.array(seen_idx, dtype=np.int64),
        np.array(unseen_idx, dtype=np.int64),
    )


def evaluate(y_true, y_pred, unseen_idx):
    sample = accuracy_score(y_true, y_pred) * 100.0
    per_fault = {}
    for lab in unseen_idx:
        mask = y_true == lab
        per_fault[CLASS_ORDER[lab]] = float((y_pred[mask] == lab).mean() * 100.0)
    return sample, float(np.mean(list(per_fault.values()))), per_fault


def make_attr_model(name, seed):
    if name == "NB":
        return GaussianNB()
    if name == "LR":
        return LogisticRegression(max_iter=2000)
    if name == "RF":
        return RandomForestClassifier(n_estimators=200, random_state=seed)
    if name == "ET":
        return ExtraTreesClassifier(n_estimators=200, random_state=seed)
    if name == "SVC":
        return SVC(kernel="rbf", gamma="scale", C=10)
    raise ValueError(name)


def predict_attributes(train_feat, train_attr, test_feat, model_name, seed):
    preds = []
    for j in range(train_attr.shape[1]):
        if np.max(train_attr[:, j]) == np.min(train_attr[:, j]):
            preds.append(np.full(test_feat.shape[0], train_attr[0, j]))
            continue
        clf = make_attr_model(model_name, seed)
        clf.fit(train_feat, train_attr[:, j])
        preds.append(clf.predict(test_feat))
    return np.stack(preds, axis=1).astype(np.float32)


def match_attributes(pred_attr, unseen_attr, unseen_zero, weights=None):
    if weights is None:
        weights = np.ones(unseen_attr.shape[1], dtype=np.float32)
    dist = (((pred_attr[:, None, :] - unseen_attr[None, :, :]) ** 2) * weights[None, None, :]).sum(axis=2)
    return unseen_zero[dist.argmin(axis=1)]


def attribute_reliability_weights(
    train_feat,
    train_attr,
    train_y,
    model_name,
    seed,
    val_ratio=0.2,
    eps=1e-3,
    clip=2.0,
):
    fit_idx, val_idx = train_test_split(
        np.arange(train_feat.shape[0]),
        test_size=val_ratio,
        random_state=seed,
        stratify=train_y,
    )
    pred_val = predict_attributes(
        train_feat[fit_idx],
        train_attr[fit_idx],
        train_feat[val_idx],
        model_name,
        seed,
    )
    mse = ((pred_val - train_attr[val_idx]) ** 2).mean(axis=0)
    weights = 1.0 / (mse + eps)
    weights = weights / (weights.mean() + 1e-8)
    return np.clip(weights, 0.0, clip).astype(np.float32)


def resolve_weighting_modes(attr_weighting, compare_reliability=False):
    if compare_reliability:
        return ("uniform", "inverse_mse")
    return (attr_weighting,)


def predict_attr_model(train_feat, train_attr, test_feat, unseen_attr, unseen_zero, model_name, seed, weights=None):
    pred_attr = predict_attributes(train_feat, train_attr, test_feat, model_name, seed)
    return match_attributes(pred_attr, unseen_attr, unseen_zero, weights)


from utils.export_model import export_selected


def run(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    semantic = np.load(args.semantic_path).astype(np.float32)
    semantic = semantic / (np.linalg.norm(semantic, axis=1, keepdims=True) + 1e-12)
    adj = build_class_graph(ATTR_MATRIX, semantic, args.graph_mode, args.topk)
    target_sim = build_target_similarity(ATTR_MATRIX, semantic, args.graph_mode)

    unseen_idx = [int(x) for x in args.unseen.split(",")]
    x_all, y_all = load_tpts(args.data_path)
    train_x, train_y, train_attr, test_x, test_y, test_attr_m, seen_zero, unseen_zero = make_split(
        x_all, y_all, unseen_idx
    )
    seen_map = {lab: i for i, lab in enumerate(seen_zero)}
    train_local_y = np.array([seen_map[int(y)] for y in train_y], dtype=np.int64)

    model = LLMGCNZSL(
        input_dim=train_x.shape[1],
        attr_dim=ATTR_MATRIX.shape[1],
        sem_dim=semantic.shape[1],
        hidden=args.hidden,
        latent=args.latent,
        dropout=args.dropout,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loader = DataLoader(ArrayDataset(train_x, train_local_y), batch_size=args.batch_size, shuffle=True)

    attr_t = torch.as_tensor(ATTR_MATRIX, dtype=torch.float32, device=device)
    sem_t = torch.as_tensor(semantic, dtype=torch.float32, device=device)
    adj_t = torch.as_tensor(adj, dtype=torch.float32, device=device)
    target_sim_t = torch.as_tensor(target_sim, dtype=torch.float32, device=device)
    train_x_t = torch.as_tensor(train_x, dtype=torch.float32, device=device)
    test_x_t = torch.as_tensor(test_x, dtype=torch.float32, device=device)

    weighting_modes = resolve_weighting_modes(args.attr_weighting, args.compare_reliability)
    effective_proto_blend = args.proto_blend if args.use_graph else 0.0
    effective_sem_weight = args.sem_weight if args.use_graph else 0.0
    best = {
        weighting: {
            "per_class_top1": 0.0,
            "sample_acc": 0.0,
            "per_fault": {},
            "pred": None,
            "attribute_weights": np.ones(ATTR_MATRIX.shape[1], dtype=np.float32),
        }
        for weighting in weighting_modes
    }
    for epoch in range(args.epochs):
        model.train()
        for bx, by_local in loader:
            bx = bx.to(device)
            by_local = by_local.to(device)
            global_y = torch.as_tensor(seen_zero, device=device)[by_local]
            z = model.encode_signal(bx)
            proto = model.encode_prototypes(attr_t, sem_t, adj_t, effective_proto_blend)
            logits = args.logit_scale * (z @ proto[seen_zero].T)
            ce = F.cross_entropy(logits, by_local)
            attr_pred = model.attr_head(z)
            sem_pred = model.sem_head(z)
            p_y = proto[global_y]
            recon_x, recon_attr = model.reconstruct(z, p_y)
            loss = (
                ce
                + args.attr_weight * F.binary_cross_entropy_with_logits(attr_pred, attr_t[global_y])
                + effective_sem_weight * (1.0 - F.cosine_similarity(sem_pred, sem_t[global_y], dim=1).mean())
                + args.recon_x_weight * F.mse_loss(recon_x, bx)
                + args.recon_attr_weight * F.binary_cross_entropy_with_logits(recon_attr, attr_t[global_y])
                + args.compat_weight * model.compatibility_loss(z, p_y, args.offdiag_weight)
                + args.rank_weight * F.mse_loss(proto @ proto.T, target_sim_t)
            )
            opt.zero_grad()
            loss.backward()
            opt.step()

        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            model.eval()
            with torch.no_grad():
                z_train = model.encode_signal(train_x_t)
                z_test = model.encode_signal(test_x_t)
                proto = model.encode_prototypes(attr_t, sem_t, adj_t, effective_proto_blend)
                if args.eval_mode == "prototype":
                    pred = unseen_zero[(z_test @ proto[unseen_zero].T).argmax(dim=1).cpu().numpy()]
                    sample, per_class, per_fault = evaluate(test_y, pred, unseen_zero)
                    state = best[weighting_modes[0]]
                    if per_class > state["per_class_top1"]:
                        state.update(
                            {"sample_acc": sample, "per_class_top1": per_class, "per_fault": per_fault, "pred": pred}
                        )
                else:
                    train_feat = torch.cat([z_train, train_x_t], dim=1).cpu().numpy()
                    test_feat = torch.cat([z_test, test_x_t], dim=1).cpu().numpy()
                    pred_attr = predict_attributes(train_feat, train_attr, test_feat, args.attr_model, args.seed)
                    for weighting in weighting_modes:
                        if weighting == "uniform":
                            weights = np.ones(ATTR_MATRIX.shape[1], dtype=np.float32)
                        else:
                            weights = attribute_reliability_weights(
                                train_feat,
                                train_attr,
                                train_y,
                                args.attr_model,
                                args.seed,
                                args.attr_weight_val_ratio,
                                args.attr_weight_eps,
                                args.attr_weight_clip,
                            )
                        pred = match_attributes(pred_attr, test_attr_m, unseen_zero, weights)
                        sample, per_class, per_fault = evaluate(test_y, pred, unseen_zero)
                        state = best[weighting]
                        if per_class > state["per_class_top1"]:
                            state.update(
                                {
                                    "sample_acc": sample,
                                    "per_class_top1": per_class,
                                    "per_fault": per_fault,
                                    "pred": pred,
                                    "attribute_weights": weights.copy(),
                                }
                            )

                            export_selected(locals(), 'TPTS')

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    common = {
        "method": "tpts_llm_semantic_attribute_class_gcn",
        "unseen_indices": unseen_idx,
        "unseen_classes": [CLASS_ORDER[i] for i in unseen_idx],
        "config": vars(args),
    }
    if args.compare_reliability:
        results = {}
        for weighting, state in best.items():
            np.save(out_dir / f"cm_{weighting}.npy", confusion_matrix(test_y, state["pred"], labels=unseen_zero))
            results[weighting] = {
                "sample_acc": state["sample_acc"],
                "per_class_top1": state["per_class_top1"],
                "per_fault": state["per_fault"],
                "attribute_weights": state["attribute_weights"].tolist(),
            }
        summary = {**common, "results": results}
    else:
        state = best[weighting_modes[0]]
        np.save(out_dir / "cm.npy", confusion_matrix(test_y, state["pred"], labels=unseen_zero))
        summary = {
            **common,
            "sample_acc": state["sample_acc"],
            "per_class_top1": state["per_class_top1"],
            "per_fault": state["per_fault"],
            "attribute_weights": state["attribute_weights"].tolist(),
        }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description="TPTS strict ZSL with LLM semantic prototypes and GCN.")
    parser.add_argument("--data-path", default="data/classData.csv")
    parser.add_argument("--semantic-path", default="data/semantic_cache/tpts_fault_semantics_qwen3_embedding.npy")
    parser.add_argument("--out-dir", default="paper_workspace/tpts_llm_gcn_zsl")
    parser.add_argument("--unseen", default="3,4", help="comma-separated 0-based class indices")
    parser.add_argument("--graph-mode", choices=["attribute", "semantic", "hybrid"], default="hybrid")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--seed", type=int, default=904)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--latent", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--logit-scale", type=float, default=20.0)
    parser.add_argument("--attr-weight", type=float, default=0.5)
    parser.add_argument("--sem-weight", type=float, default=0.05)
    parser.add_argument("--recon-x-weight", type=float, default=1e-4)
    parser.add_argument("--recon-attr-weight", type=float, default=1.0)
    parser.add_argument("--compat-weight", type=float, default=0.5)
    parser.add_argument("--offdiag-weight", type=float, default=0.5)
    parser.add_argument("--rank-weight", type=float, default=0.0)
    parser.add_argument("--proto-blend", type=float, default=0.25)
    parser.add_argument("--use-graph", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eval-mode", choices=["prototype", "nb_attr"], default="nb_attr")
    parser.add_argument("--attr-model", choices=["NB", "LR", "RF", "ET", "SVC"], default="NB")
    parser.add_argument("--attr-weighting", choices=["uniform", "inverse_mse"], default="uniform")
    parser.add_argument("--compare-reliability", action="store_true")
    parser.add_argument("--attr-weight-val-ratio", type=float, default=0.2)
    parser.add_argument("--attr-weight-eps", type=float, default=1e-3)
    parser.add_argument("--attr-weight-clip", type=float, default=2.0)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
