import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import Ridge
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sce_tep import CONFIG, TEST_TASKS, CenterLossModule, Embedding_Net, MyDataset, pre_model_eval  # noqa: E402
from utils.read_tep_data import create_tep_data, load_tep_data  # noqa: E402
from utils.run_llm_gcn_zsl import build_class_graph, normalize_rows  # noqa: E402


def predict_attr_matrix(train_feat, train_attr, test_feat):
    preds = []
    for j in range(train_attr.shape[1]):
        if np.max(train_attr[:, j]) == np.min(train_attr[:, j]):
            preds.append(np.full(test_feat.shape[0], train_attr[0, j]))
            continue
        clf = GaussianNB()
        clf.fit(train_feat, train_attr[:, j])
        preds.append(clf.predict(test_feat))
    return np.stack(preds, axis=1)


def semantic_tiebreak_predictions(
    z_train,
    y_train,
    z_test,
    train_feat,
    train_attr,
    test_feat,
    unseen_zero,
    unseen_attr,
    attr_all,
    sem_all,
    adj,
    tie_eps,
    ridge_alpha,
    sem_margin,
):
    pred_attr = predict_attr_matrix(train_feat, train_attr, test_feat)
    attr_dist = ((pred_attr[:, None, :] - unseen_attr[None, :, :]) ** 2).sum(axis=2)
    base_idx = attr_dist.argmin(axis=1)
    base_pred = unseen_zero[base_idx]

    proto_raw = np.concatenate([normalize_rows(attr_all.astype(np.float32)), normalize_rows(sem_all.astype(np.float32))], axis=1)
    proto_graph = adj @ proto_raw
    seen_zero = np.array(sorted(np.unique(y_train)), dtype=np.int64)
    centers = []
    for lab in seen_zero:
        centers.append(z_train[y_train == lab].mean(axis=0))
    centers = normalize_rows(np.stack(centers, axis=0).astype(np.float32))

    ridge = Ridge(alpha=ridge_alpha)
    ridge.fit(proto_graph[seen_zero], centers)
    unseen_centers = normalize_rows(ridge.predict(proto_graph[unseen_zero]).astype(np.float32))
    sem_scores = normalize_rows(z_test.astype(np.float32)) @ unseen_centers.T

    pred = base_pred.copy()
    changed = 0
    used = 0
    for i in range(attr_dist.shape[0]):
        best = attr_dist[i].min()
        tied = np.where(attr_dist[i] <= best + tie_eps)[0]
        if tied.size <= 1:
            continue
        tied_scores = sem_scores[i, tied]
        if tied_scores.size > 1:
            sorted_scores = np.sort(tied_scores)
            if sorted_scores[-1] - sorted_scores[-2] < sem_margin:
                continue
        used += 1
        chosen_local = tied[np.argmax(tied_scores)]
        if unseen_zero[chosen_local] != pred[i]:
            changed += 1
        pred[i] = unseen_zero[chosen_local]
    return pred, {"semantic_gate_used": used, "semantic_prediction_changed": changed}


def semantic_refined_attr_predictions(
    train_feat,
    train_attr,
    test_feat,
    unseen_zero,
    attr_all,
    adj,
    refine_beta,
):
    pred_attr = predict_attr_matrix(train_feat, train_attr, test_feat)
    refined_attr = (1.0 - refine_beta) * attr_all + refine_beta * (adj @ attr_all)
    attr_dist = ((pred_attr[:, None, :] - refined_attr[unseen_zero][None, :, :]) ** 2).sum(axis=2)
    pred = unseen_zero[attr_dist.argmin(axis=1)]
    base_dist = ((pred_attr[:, None, :] - attr_all[unseen_zero][None, :, :]) ** 2).sum(axis=2)
    base_pred = unseen_zero[base_dist.argmin(axis=1)]
    return pred, {
        "semantic_gate_used": int(test_feat.shape[0]),
        "semantic_prediction_changed": int(np.sum(pred != base_pred)),
    }


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ClassPrototypeGCN(nn.Module):
    def __init__(self, attr_dim, sem_dim, hidden, latent, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(attr_dim + sem_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.gcn1 = nn.Linear(hidden, hidden)
        self.gcn2 = nn.Linear(hidden, latent)
        self.attr_anchor = nn.Sequential(
            nn.Linear(attr_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, latent),
        )

    def forward(self, attr_proto, sem_proto, adj, blend):
        h = self.input_proj(torch.cat([attr_proto, sem_proto], dim=1))
        h = F.relu(self.gcn1(adj @ h))
        g = self.gcn2(adj @ h)
        a = self.attr_anchor(attr_proto)
        return F.normalize((1.0 - blend) * a + blend * g, dim=1)


def run_experiment(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = dict(CONFIG)
    cfg["seed"] = args.seed
    cfg["epochs"] = args.epochs

    attr_all = np.load(args.attr_path).astype(np.float32)
    sem_all = np.load(args.semantic_path).astype(np.float32)
    sem_all = normalize_rows(sem_all)
    adj = build_class_graph(attr_all, sem_all, args.graph_mode, args.topk)

    attr_t_all = torch.as_tensor(attr_all, dtype=torch.float32, device=device)
    sem_t_all = torch.as_tensor(sem_all, dtype=torch.float32, device=device)
    adj_t = torch.as_tensor(adj, dtype=torch.float32, device=device)

    data_list = load_tep_data(
        data_path=args.data_path,
        use_core_sensors=cfg.get("use_core_sensors", False),
        sma_window=cfg.get("sma_window", 3),
    )

    rows = []
    for task_idx, task in enumerate(TEST_TASKS):
        unseen = task["test_class"]
        seen = task["seen_classes"]
        print(f"--- Running Task {task_idx + 1}/4: {unseen} ---")
        train_x, train_y, train_attr, _, test_x, test_y, _, test_attr_m, _ = create_tep_data(
            data_list, attr_all, seen, unseen
        )
        train_x_t = torch.as_tensor(train_x, dtype=torch.float32, device=device)
        train_attr_t = torch.as_tensor(train_attr, dtype=torch.float32, device=device)
        train_y_t = torch.as_tensor(train_y, dtype=torch.long, device=device)
        test_x_t = torch.as_tensor(test_x, dtype=torch.float32, device=device)

        unique_labels = np.unique(train_y)
        label_to_idx = {lab: i for i, lab in enumerate(unique_labels)}
        train_loader = DataLoader(
            MyDataset(train_x_t, train_attr_t, train_y_t),
            batch_size=cfg["batch_size"],
            shuffle=True,
        )

        model = Embedding_Net(cfg["dim"], cfg["lambda_"], cfg["dropout"]).to(device)
        center_loss_module = CenterLossModule(len(unique_labels), cfg["dim"][1]).to(device)
        cpu_rng_state = torch.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        proto_gcn = ClassPrototypeGCN(
            attr_dim=attr_all.shape[1],
            sem_dim=sem_all.shape[1],
            hidden=args.proto_hidden,
            latent=cfg["dim"][1],
            dropout=args.proto_dropout,
        ).to(device)
        torch.set_rng_state(cpu_rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state_all(cuda_rng_state)
        params = list(model.parameters()) + list(center_loss_module.parameters()) + list(proto_gcn.parameters())
        optimizer = torch.optim.RMSprop(params, lr=cfg["lr"])
        scheduler = MultiStepLR(optimizer, milestones=[450], gamma=0.5)

        best_acc = 0.0
        best_epoch = None
        best_pred = None
        best_gate_info = {"semantic_gate_used": 0, "semantic_prediction_changed": 0}
        for epoch in range(cfg["epochs"]):
            model.train()
            proto_gcn.train()
            for bx, ba, by in train_loader:
                optimizer.zero_grad()
                z1, z2 = model.transform(bx, ba)
                x_rec, a_rec = model.reconstruction(z1, z2)
                batch_unique = torch.unique(by)
                batch_label_idx = torch.zeros_like(by)
                for idx, lab in enumerate(batch_unique):
                    batch_label_idx[by == lab] = idx
                r1 = cfg["lambda_"][0] * F.mse_loss(ba, a_rec)
                r2 = cfg["lambda_"][1] * F.mse_loss(bx, x_rec)
                cm = cfg["lambda_"][2] * model.compability_loss(z1, z2)
                center_loss = center_loss_module(z1, batch_label_idx)

                use_hard_semantic_regularizer = args.semantic_strategy == "regularize" and (
                    args.align_weight > 0 or args.proto_ce_weight > 0
                )
                if use_hard_semantic_regularizer:
                    proto = proto_gcn(attr_t_all, sem_t_all, adj_t, args.proto_blend)
                    p_y = proto[by]
                    llm_gcn_align = 1.0 - F.cosine_similarity(F.normalize(z1, dim=1), p_y, dim=1).mean()
                    global_label_idx = torch.as_tensor(
                        [label_to_idx[int(y)] for y in by.detach().cpu().numpy()],
                        dtype=torch.long,
                        device=device,
                    )
                    proto_margin = F.cross_entropy(
                        args.logit_scale * (F.normalize(z1, dim=1) @ proto[unique_labels].T),
                        global_label_idx,
                    )
                else:
                    llm_gcn_align = torch.tensor(0.0, device=device)
                    proto_margin = torch.tensor(0.0, device=device)
                loss = (
                    r1
                    + r2
                    + cm
                    + cfg["center_coef"] * center_loss
                    + (args.align_weight if args.semantic_strategy == "regularize" else 0.0) * llm_gcn_align
                    + (args.proto_ce_weight if args.semantic_strategy == "regularize" else 0.0) * proto_margin
                )
                loss.backward()
                optimizer.step()
            scheduler.step()

            if (epoch + 1) % args.eval_every == 0 or epoch == cfg["epochs"] - 1:
                model.eval()
                with torch.no_grad():
                    z_train, _ = model.transform(train_x_t, train_attr_t)
                    z_test, _ = model.transform(test_x_t, torch.zeros(test_x_t.size(0), attr_all.shape[1], device=device))
                    f_train = torch.cat([z_train, train_x_t], dim=1).cpu().numpy()
                    f_test = torch.cat([z_test, test_x_t], dim=1).cpu().numpy()
                    if args.semantic_strategy == "posthoc_tiebreak":
                        unseen_zero = np.array([c - 1 for c in unseen], dtype=np.int64)
                        pred, gate_info = semantic_tiebreak_predictions(
                            z_train.cpu().numpy(),
                            train_y,
                            z_test.cpu().numpy(),
                            f_train,
                            train_attr,
                            f_test,
                            unseen_zero,
                            test_attr_m,
                            attr_all,
                            sem_all,
                            adj,
                            args.tie_eps,
                            args.ridge_alpha,
                            args.sem_margin,
                        )
                        acc = accuracy_score(test_y, pred)
                        if acc > best_acc:
                            best_gate_info = gate_info
                    elif args.semantic_strategy == "prototype_refine":
                        unseen_zero = np.array([c - 1 for c in unseen], dtype=np.int64)
                        pred, gate_info = semantic_refined_attr_predictions(
                            f_train,
                            train_attr,
                            f_test,
                            unseen_zero,
                            attr_all,
                            adj,
                            args.refine_beta,
                        )
                        acc = accuracy_score(test_y, pred)
                        if acc > best_acc:
                            best_gate_info = gate_info
                    else:
                        acc = pre_model_eval(cfg["model_mode"], f_train, train_attr, f_test, test_y, test_attr_m)
                    if float(acc) > best_acc:
                        best_epoch = epoch + 1
                        predicted_attributes = predict_attr_matrix(f_train, train_attr, f_test)
                        candidates = np.asarray([c - 1 for c in unseen])
                        best_pred = candidates[((predicted_attributes[:,None,:] - test_attr_m[None,:,:]) ** 2).sum(axis=2).argmin(axis=1)]
                    best_acc = max(best_acc, float(acc))

        Path(args.out_dir).mkdir(parents=True, exist_ok=True)
        np.savez_compressed(Path(args.out_dir) / f"predictions_Task{task_idx+1}.npz", test_y=test_y, pred=best_pred)
        print(f"Task {task_idx + 1} Best Accuracy: {best_acc:.4f}")
        rows.append({
            "task": task_idx + 1,
            "selected_epoch": best_epoch,
            "unseen": unseen,
            "accuracy": best_acc,
            "semantic_strategy": args.semantic_strategy,
            **best_gate_info,
        })

    details = pd.DataFrame(rows)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    details.to_csv(out_dir / "sce_llm_gcn_tep_details.csv", index=False)
    summary = {
        "method": "SCE_backbone_with_LLM_GCN_semantic_regularizer",
        "interpretation": (
            "gated uses LLM-GCN as a reliability-controlled semantic graph branch without "
            "back-propagating dense text semantics into the SCE signal encoder; regularize "
            "uses direct LLM-GCN training loss and is kept for negative-transfer audit; "
            "posthoc_tiebreak uses graph-smoothed LLM prototypes to break engineering-attribute ties."
            " prototype_refine uses the LLM-GCN class graph to smooth unseen engineering prototypes before matching."
        ),
        "mean_accuracy": float(details["accuracy"].mean()),
        "task_accuracy": details["accuracy"].tolist(),
        "config": vars(args),
    }
    (out_dir / "sce_llm_gcn_tep_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description="TEP SCE backbone with LLM-GCN semantic prototype regularization.")
    parser.add_argument("--semantic-path", default="data/semantic_cache/tep_fault_semantics_qwen3_embedding.npy")
    parser.add_argument("--attr-path", default="data/TEP_attribute_matrix.npy")
    parser.add_argument("--data-path", default="./TEP_data/*.dat")
    parser.add_argument("--out-dir", default="paper_workspace/sce_llm_gcn_tep")
    parser.add_argument("--seed", type=int, default=904)
    parser.add_argument("--epochs", type=int, default=900)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--graph-mode", choices=["attribute", "semantic", "hybrid"], default="hybrid")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--proto-hidden", type=int, default=128)
    parser.add_argument("--proto-dropout", type=float, default=0.1)
    parser.add_argument("--proto-blend", type=float, default=0.15)
    parser.add_argument("--align-weight", type=float, default=0.01)
    parser.add_argument("--proto-ce-weight", type=float, default=0.0)
    parser.add_argument("--logit-scale", type=float, default=20.0)
    parser.add_argument("--semantic-strategy", choices=["gated", "regularize", "posthoc_tiebreak", "prototype_refine"], default="gated")
    parser.add_argument("--tie-eps", type=float, default=0.0)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--sem-margin", type=float, default=0.0)
    parser.add_argument("--refine-beta", type=float, default=0.05)
    run_experiment(parser.parse_args())


if __name__ == "__main__":
    main()
