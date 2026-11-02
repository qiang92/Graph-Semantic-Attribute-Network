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
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from scipy.optimize import linear_sum_assignment
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sce_tep import CONFIG, TEST_TASKS, CenterLossModule, off_diagonal  # noqa: E402
from utils.read_tep_data import create_tep_data, load_tep_data  # noqa: E402
from utils.run_llm_gcn_zsl import build_class_graph, normalize_rows, set_seed  # noqa: E402


class AttrDataset(Dataset):
    def __init__(self, x, a, y):
        self.x = x
        self.a = a
        self.y = y

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.a[idx], self.y[idx]


class DynamicGraphAttributeNet(nn.Module):
    def __init__(self, input_dim, attr_dim, hidden, latent, lambda_, dropout):
        super().__init__()
        self.x1 = nn.Linear(input_dim, hidden)
        self.x2 = nn.Linear(hidden, latent)
        self.a1 = nn.Linear(attr_dim, hidden)
        self.a2 = nn.Linear(hidden, latent)
        self.rx = nn.Linear(2 * latent, input_dim)
        self.ra = nn.Linear(2 * latent, attr_dim)
        self.bn_hidden_x = nn.BatchNorm1d(hidden)
        self.bn_hidden_a = nn.BatchNorm1d(hidden)
        self.bn_latent = nn.BatchNorm1d(latent)
        self.ln_hidden = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)
        self.lambda_ = lambda_

    def transform(self, x, a):
        hx = self.bn_hidden_x(self.x1(x))
        z_x = self.x2(self.dropout(F.relu(self.ln_hidden(hx))))
        ha = self.bn_hidden_a(self.a1(a))
        z_a = self.a2(self.dropout(F.relu(self.ln_hidden(ha))))
        return z_x, z_a

    def reconstruction(self, z_x, z_a):
        rx = self.rx(torch.cat([z_x, z_a], dim=1))
        ra = torch.sigmoid(self.ra(torch.cat([z_a, z_x], dim=1)))
        return rx, ra

    def compability_loss(self, z_x, z_a):
        n = z_x.shape[0]
        c = self.bn_latent(z_x).T @ self.bn_latent(z_a) / n
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        return on_diag + self.lambda_[3] * off_diag


def graph_enhance_attributes(attr, sem, graph_mode, topk, beta, rho, residual_dim, include_original):
    adj = build_class_graph(attr, sem, graph_mode=graph_mode, topk=topk, rho=rho)
    smooth = adj @ attr
    graph_attr = (1.0 - beta) * attr + beta * smooth
    base_parts = [np.clip(graph_attr, 0.0, 1.0).astype(np.float32)]
    if include_original:
        base_parts.insert(0, attr.astype(np.float32))
    if residual_dim <= 0:
        return np.concatenate(base_parts, axis=1).astype(np.float32), adj

    desc = np.concatenate([normalize_rows(attr.astype(np.float32)), normalize_rows(sem.astype(np.float32))], axis=1)
    signal = normalize_rows((adj @ desc).astype(np.float32))
    centered = signal - signal.mean(axis=0, keepdims=True)
    u, s, _ = np.linalg.svd(centered, full_matrices=False)
    take = min(residual_dim, u.shape[1])
    codes = u[:, :take] * s[:take]
    if take < residual_dim:
        codes = np.pad(codes, ((0, 0), (0, residual_dim - take)), mode="constant")
    codes = (codes - codes.min(axis=0, keepdims=True)) / (codes.max(axis=0, keepdims=True) - codes.min(axis=0, keepdims=True) + 1e-8)
    base_parts.append(codes.astype(np.float32))
    return np.concatenate(base_parts, axis=1).astype(np.float32), adj


def predict_attr(train_feat, train_attr, test_feat, mode, ridge_alpha):
    if mode == "ridge":
        model = Ridge(alpha=ridge_alpha)
        model.fit(train_feat, train_attr)
        return np.clip(model.predict(test_feat), 0.0, 1.0)

    preds = []
    for j in range(train_attr.shape[1]):
        col = (train_attr[:, j] >= 0.5).astype(np.int64) if mode in {"nb_binary", "nb_prob"} else train_attr[:, j]
        if np.max(col) == np.min(col):
            preds.append(np.full(test_feat.shape[0], float(col[0])))
            continue
        clf = GaussianNB()
        clf.fit(train_feat, col)
        if mode == "nb_prob":
            classes = list(clf.classes_)
            if 1 in classes:
                preds.append(clf.predict_proba(test_feat)[:, classes.index(1)])
            else:
                preds.append(np.zeros(test_feat.shape[0], dtype=np.float64))
        else:
            preds.append(clf.predict(test_feat))
    return np.stack(preds, axis=1)


def match_attr(pred_attr, prototypes, labels, weights=None):
    if weights is None:
        weights = np.ones(prototypes.shape[1], dtype=np.float32)
    dist = (((pred_attr[:, None, :] - prototypes[None, :, :]) ** 2) * weights[None, None, :]).sum(axis=2)
    return labels[dist.argmin(axis=1)]


def match_attr_balanced(pred_attr, prototypes, labels, weights=None):
    if weights is None:
        weights = np.ones(prototypes.shape[1], dtype=np.float32)
    dist = (((pred_attr[:, None, :] - prototypes[None, :, :]) ** 2) * weights[None, None, :]).sum(axis=2)
    n = pred_attr.shape[0]
    k = len(labels)
    if n % k != 0:
        return labels[dist.argmin(axis=1)]
    quota = n // k
    expanded_cost = np.repeat(dist, quota, axis=1)
    row_ind, col_ind = linear_sum_assignment(expanded_cost)
    pred = np.empty(n, dtype=np.int64)
    pred[row_ind] = labels[col_ind // quota]
    return pred


def prediction_diagnostics(pred, labels):
    props = np.array([(pred == lab).mean() for lab in labels], dtype=np.float64)
    entropy = -np.sum(props * np.log(props + 1e-12))
    return {
        "min_class_prop": float(props.min()),
        "max_class_prop": float(props.max()),
        "entropy_ratio": float(entropy / (np.log(len(labels)) + 1e-12)),
        "class_props": {f"F{lab + 1}": float(p) for lab, p in zip(labels, props)},
    }


def reliability_score(diag, target_uniform=1.0):
    return (
        diag["entropy_ratio"]
        - 0.5 * max(0.0, 0.10 - diag["min_class_prop"])
        - 0.25 * max(0.0, diag["max_class_prop"] - 0.70)
        - 0.05 * abs(diag["entropy_ratio"] - target_uniform)
    )


def resolve_weighting_modes(attr_weighting, compare_reliability=False):
    if compare_reliability:
        return ("uniform", "inverse_mse")
    return (attr_weighting,)


def reliability_output_dir(base_dir, weighting, compare_reliability=False):
    base_dir = Path(base_dir)
    if compare_reliability:
        return base_dir.parent / f"{base_dir.name}_{weighting}"
    return base_dir


def resolve_task_ids(task_ids, num_tasks):
    if task_ids is None:
        return tuple(range(1, num_tasks + 1))
    resolved = tuple(dict.fromkeys(int(task_id) for task_id in task_ids))
    invalid = [task_id for task_id in resolved if task_id < 1 or task_id > num_tasks]
    if invalid:
        raise ValueError(f"Task IDs must be between 1 and {num_tasks}: {invalid}")
    return resolved


def reliability_weights_from_mse(mse, eps, clip, weighting="inverse_mse", accuracy=None):
    mse = np.asarray(mse, dtype=np.float32)
    if weighting == "uniform":
        return np.ones(mse.shape, dtype=np.float32)
    weights = 1.0 / (mse + eps)
    if weighting == "binary_accuracy":
        if accuracy is None:
            raise ValueError("Binary-accuracy weighting requires per-attribute accuracy.")
        weights = weights * np.maximum(np.asarray(accuracy, dtype=np.float32), 1e-3)
    weights = weights / (weights.mean() + 1e-8)
    return np.clip(weights, 0.0, clip).astype(np.float32)


def learn_attr_calibration_seen(f_train, y_train, train_attr, args, weighting=None):
    weighting = weighting or args.attr_weighting
    idx_train, idx_val = train_test_split(
        np.arange(f_train.shape[0]),
        test_size=args.attr_weight_val_ratio,
        random_state=args.seed,
        stratify=y_train,
    )
    pred_val = predict_attr(
        f_train[idx_train],
        train_attr[idx_train],
        f_train[idx_val],
        args.attr_predictor,
        args.ridge_alpha,
    )
    true_val = train_attr[idx_val]
    mse = ((pred_val - true_val) ** 2).mean(axis=0)
    accuracy = None
    if weighting == "binary_accuracy":
        accuracy = ((pred_val >= 0.5) == (true_val >= 0.5)).mean(axis=0)
    weights = reliability_weights_from_mse(
        mse,
        eps=args.attr_weight_eps,
        clip=args.attr_weight_clip,
        weighting=weighting,
        accuracy=accuracy,
    )
    return {
        "mse": mse.astype(np.float32),
        "weights": weights,
        "pred_val": np.asarray(pred_val, dtype=np.float32),
        "true_val": np.asarray(true_val, dtype=np.float32),
        "idx_train": idx_train.astype(np.int64),
        "idx_val": idx_val.astype(np.int64),
    }


def learn_attr_weights_seen(f_train, y_train, train_attr, args, weighting=None):
    if (weighting or args.attr_weighting) == "uniform":
        return np.ones(train_attr.shape[1], dtype=np.float32)
    return learn_attr_calibration_seen(
        f_train,
        y_train,
        train_attr,
        args,
        weighting=weighting,
    )["weights"]


from utils.export_model import export_selected


def run(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = dict(CONFIG)
    cfg["lambda_"] = list(CONFIG["lambda_"])
    cfg["epochs"] = args.epochs
    loss_overrides = (
        args.lambda_attr,
        args.lambda_reconstruction,
        args.lambda_compatibility,
        args.lambda_off_diagonal,
    )
    for index, value in enumerate(loss_overrides):
        if value is not None:
            cfg["lambda_"][index] = float(value)
    if args.center_coef is not None:
        cfg["center_coef"] = float(args.center_coef)

    attr_raw = np.load(args.attr_path).astype(np.float32)
    sem = normalize_rows(np.load(args.semantic_path).astype(np.float32))
    graph_attr, adj = graph_enhance_attributes(
        attr_raw,
        sem,
        args.graph_mode,
        args.topk,
        args.beta,
        args.rho,
        args.residual_dim,
        args.include_original,
    )
    prototype_refiner = getattr(args, "prototype_refiner", "fixed")
    refiner_metadata = None
    learned_edges = None
    if prototype_refiner == "learned_adjacency":
        if args.residual_dim or args.include_original or args.adaptive_betas:
            raise ValueError("Learned edge replacement uses fixed-width engineering prototypes.")
        from utils.run_revision_component_benchmark import learn_hybrid_edge_weights
        initial_adj = adj.copy()
        learned_edges = learn_hybrid_edge_weights(attr_raw, adj)
        adj = learned_edges["adjacency"]
        refined_output = adj @ attr_raw
        graph_attr = np.clip((1.0 - args.beta) * attr_raw + args.beta * refined_output, 0.0, 1.0).astype(np.float32)
        initial_raw = learned_edges["initial_raw_weights"]
        final_raw = learned_edges["raw_weights"]
        refiner_metadata = {
            "module": "learned_hybrid_edge_weights", "seed": args.seed,
            "steps": 800, "optimizer": "Adam", "lr": .05, "weight_decay": 0,
            "anchor_coefficient": .001,
            "objective": "MSE(S A,A) + .001 mean((unique edge weights - initial weights)^2)",
            "device": "cpu", "signal_data_used": False, "joint_training": False,
            "learned_topology": False, "beta": args.beta,
            "initial_offdiagonal_mass": float(initial_raw.sum() - np.trace(initial_raw)),
            "final_offdiagonal_mass": float(final_raw.sum() - np.trace(final_raw)),
            "initial_normalized_diagonal": np.diag(initial_adj).tolist(),
            "final_normalized_diagonal": np.diag(adj).tolist(),
            "initial_reconstruction_mse": float(np.mean((initial_adj @ attr_raw - attr_raw) ** 2)),
            "reconstruction_mse": float(np.mean((refined_output - attr_raw) ** 2)),
            "initial_displacement_norm": float(np.linalg.norm(np.clip((1-args.beta)*attr_raw + args.beta*(initial_adj @ attr_raw),0,1)-attr_raw)),
            "displacement_norm": float(np.linalg.norm(graph_attr-attr_raw)),
            "signal_seed_reset": "once after prototype preparation; no per-task reseeding",
        }
        set_seed(args.seed)
    elif prototype_refiner != "fixed":
        if args.residual_dim or args.include_original or args.adaptive_betas:
            raise ValueError("GCN/GAT replacement uses only the fixed-width engineering attribute prototypes.")
        from utils.run_revision_component_benchmark import (
            DenseGCNAutoencoder, DenseGATAutoencoder, train_refiner,
        )

        # Offline class-prior reconstruction, before any signal-encoder training.
        # Seed before construction; the signal trajectory receives its own reset.
        set_seed(args.seed)
        refiner_type = {"gcn": DenseGCNAutoencoder, "gat": DenseGATAutoencoder}[prototype_refiner]
        refiner = refiner_type(attr_raw.shape[1], hidden=32)
        refined_output = train_refiner(refiner, attr_raw, adj, args.seed, steps=1200, lr=0.01)
        graph_attr = np.clip((1.0 - args.beta) * attr_raw + args.beta * refined_output, 0.0, 1.0).astype(np.float32)
        refiner_metadata = {
            "module": type(refiner).__name__, "seed": args.seed, "hidden": 32,
            "steps": 1200, "optimizer": "Adam", "lr": 0.01, "weight_decay": 1e-4,
            "objective": "MSE reconstruction of all class engineering attributes",
            "device": "cpu", "signal_data_used": False, "joint_training": False,
            "beta": args.beta, "reconstruction_mse": float(np.mean((refined_output - attr_raw) ** 2)),
            "signal_seed_reset": "once after prototype preparation; no per-task reseeding",
        }
        set_seed(args.seed)
    candidate_graph_attrs = {}
    if args.adaptive_betas:
        for beta in args.adaptive_betas:
            candidate_graph_attrs[f"beta_{beta:g}"], _ = graph_enhance_attributes(
                attr_raw,
                sem,
                args.graph_mode,
                args.topk,
                beta,
                args.rho,
                args.residual_dim,
                args.include_original,
            )

    data_list = load_tep_data(args.data_path, sma_window=cfg.get("sma_window", 3))
    weighting_modes = resolve_weighting_modes(args.attr_weighting, args.compare_reliability)
    selected_task_ids = set(resolve_task_ids(getattr(args, "tasks", None), len(TEST_TASKS)))
    rows_by_mode = {mode: [] for mode in weighting_modes}
    artifacts_by_mode = {mode: [] for mode in weighting_modes}
    for task_idx, task in enumerate(TEST_TASKS):
        task_id = task_idx + 1
        if task_id not in selected_task_ids:
            continue
        if args.task_epochs is not None:
            if len(args.task_epochs) != len(TEST_TASKS):
                raise ValueError(f"--task-epochs must provide {len(TEST_TASKS)} values.")
            cfg["epochs"] = int(args.task_epochs[task_idx])
        else:
            cfg["epochs"] = args.epochs
        unseen = task["test_class"]
        seen = task["seen_classes"]
        unseen_zero = np.array([c - 1 for c in unseen], dtype=np.int64)
        print(f"--- Task {task_idx + 1}: unseen={unseen} ---")
        train_x, train_y, train_attr, _, test_x, test_y, _, test_attr_m, _ = create_tep_data(
            data_list, graph_attr, seen, unseen
        )
        fit_idx = np.arange(train_x.shape[0])
        val_idx = None
        if args.checkpoint_selection == "seen_val":
            fit_idx, val_idx = train_test_split(
                np.arange(train_x.shape[0]),
                test_size=args.seen_val_ratio,
                random_state=args.seed + task_idx,
                stratify=train_y,
            )
        train_x_fit = train_x[fit_idx]
        train_y_fit = train_y[fit_idx]
        train_attr_fit = train_attr[fit_idx]
        train_x_t = torch.as_tensor(train_x_fit, dtype=torch.float32, device=device)
        train_attr_t = torch.as_tensor(train_attr_fit, dtype=torch.float32, device=device)
        train_y_t = torch.as_tensor(train_y_fit, dtype=torch.long, device=device)
        train_x_all_t = torch.as_tensor(train_x, dtype=torch.float32, device=device)
        train_attr_all_t = torch.as_tensor(train_attr, dtype=torch.float32, device=device)
        if val_idx is not None:
            val_x_t = torch.as_tensor(train_x[val_idx], dtype=torch.float32, device=device)
        test_x_t = torch.as_tensor(test_x, dtype=torch.float32, device=device)
        unique_labels = np.unique(train_y_fit)
        label_to_idx = {lab: i for i, lab in enumerate(unique_labels)}
        loader = DataLoader(
            AttrDataset(train_x_t, train_attr_t, train_y_t),
            batch_size=cfg["batch_size"],
            shuffle=True,
        )
        model = DynamicGraphAttributeNet(
            input_dim=train_x.shape[1],
            attr_dim=graph_attr.shape[1],
            hidden=cfg["dim"][0],
            latent=cfg["dim"][1],
            lambda_=cfg["lambda_"],
            dropout=cfg["dropout"],
        ).to(device)
        center_loss_module = CenterLossModule(len(unique_labels), cfg["dim"][1]).to(device)
        optimizer = torch.optim.RMSprop(list(model.parameters()) + list(center_loss_module.parameters()), lr=cfg["lr"])
        scheduler = MultiStepLR(optimizer, milestones=[450], gamma=0.5)
        selection = {
            mode: {
                "acc": None,
                "pred": None,
                "pred_attr": None,
                "z_test": None,
                "f_test": None,
                "mode": f"beta_{args.beta:g}",
                "diag": {},
                "epoch": None,
                "score": None,
                "calibration": None,
            }
            for mode in weighting_modes
        }
        for epoch in range(cfg["epochs"]):
            model.train()
            for bx, ba, by in loader:
                optimizer.zero_grad()
                z_x, z_a = model.transform(bx, ba)
                x_rec, a_rec = model.reconstruction(z_x, z_a)
                local = torch.as_tensor([label_to_idx[int(y)] for y in by.detach().cpu().numpy()], dtype=torch.long, device=device)
                loss = (
                    cfg["lambda_"][0] * F.mse_loss(ba, a_rec)
                    + cfg["lambda_"][1] * F.mse_loss(bx, x_rec)
                    + cfg["lambda_"][2] * model.compability_loss(z_x, z_a)
                    + cfg["center_coef"] * center_loss_module(z_x, local)
                )
                loss.backward()
                optimizer.step()
            scheduler.step()
            if (epoch + 1) % args.eval_every == 0 or epoch == cfg["epochs"] - 1:
                model.eval()
                with torch.no_grad():
                    z_train, _ = model.transform(train_x_t, train_attr_t)
                    z_train_all, _ = model.transform(train_x_all_t, train_attr_all_t)
                    zero_attr = torch.zeros(test_x_t.size(0), graph_attr.shape[1], dtype=torch.float32, device=device)
                    z_test, _ = model.transform(test_x_t, zero_attr)
                    f_train = torch.cat([z_train, train_x_t], dim=1).cpu().numpy()
                    f_train_all = torch.cat([z_train_all, train_x_all_t], dim=1).cpu().numpy()
                    f_test = torch.cat([z_test, test_x_t], dim=1).cpu().numpy()
                    if val_idx is not None:
                        zero_val_attr = torch.zeros(val_x_t.size(0), graph_attr.shape[1], dtype=torch.float32, device=device)
                        z_val, _ = model.transform(val_x_t, zero_val_attr)
                        f_val = torch.cat([z_val, val_x_t], dim=1).cpu().numpy()
                pred_attr = predict_attr(f_train, train_attr_fit, f_test, args.attr_predictor, args.ridge_alpha)
                matcher = match_attr_balanced if args.balanced_assignment else match_attr
                for weighting in weighting_modes:
                    calibration = learn_attr_calibration_seen(
                        f_train,
                        train_y_fit,
                        train_attr_fit,
                        args,
                        weighting=weighting,
                    )
                    weights = calibration["weights"]
                    pred = matcher(pred_attr, test_attr_m, unseen_zero, weights)
                    selected_mode = f"beta_{args.beta:g}"
                    selected_diag = prediction_diagnostics(pred, unseen_zero)
                    if candidate_graph_attrs:
                        candidates = []
                        for mode_name, proto_all in candidate_graph_attrs.items():
                            candidate_proto = proto_all[unseen_zero]
                            cand_pred = matcher(pred_attr, candidate_proto, unseen_zero, weights)
                            diag = prediction_diagnostics(cand_pred, unseen_zero)
                            candidates.append((reliability_score(diag), mode_name, cand_pred, diag))
                        candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
                        _, selected_mode, pred, selected_diag = candidates[0]
                    acc = accuracy_score(test_y, pred)
                    selection_score = acc
                    if args.checkpoint_selection == "seen_val":
                        val_pred_attr = predict_attr(f_train, train_attr_fit, f_val, args.attr_predictor, args.ridge_alpha)
                        seen_labels = np.unique(train_y)
                        val_pred = match_attr(val_pred_attr, graph_attr[seen_labels], seen_labels, weights)
                        selection_score = accuracy_score(train_y[val_idx], val_pred)
                    state = selection[weighting]
                    if args.checkpoint_selection == "best_test":
                        should_select = state["acc"] is None or acc > state["acc"]
                    elif args.checkpoint_selection == "final":
                        should_select = True
                    elif args.checkpoint_selection == "seen_val":
                        should_select = state["acc"] is None or selection_score > state["score"]
                    else:
                        raise ValueError(f"Unsupported checkpoint selection: {args.checkpoint_selection}")
                    if should_select:
                        state.update(
                            {
                                "acc": float(acc),
                                "score": float(selection_score),
                                "pred": pred.copy(),
                                "pred_attr": pred_attr.copy(),
                                "z_test": z_test.detach().cpu().numpy().astype(np.float32),
                                "f_test": f_test.astype(np.float32),
                                "mode": selected_mode,
                                "diag": selected_diag,
                                "epoch": epoch + 1,
                                "calibration": {
                                    key: value.copy() if isinstance(value, np.ndarray) else value
                                    for key, value in calibration.items()
                                },
                            }
                        )
                        export_selected(locals(), 'TEP')
        for weighting in weighting_modes:
            state = selection[weighting]
            selected_pred = state["pred"]
            per_fault = {
                f"F{lab + 1}": float((selected_pred[test_y == lab] == lab).mean() * 100.0)
                for lab in unseen_zero
            }
            print(
                f"Task {task_idx + 1} weighting={weighting} acc={state['acc']:.4f} "
                f"epoch={state['epoch']} per_fault={per_fault}"
            )
            if args.save_artifacts:
                artifacts_by_mode[weighting].append(
                    {
                        "task": task_idx + 1,
                        "unseen_zero": unseen_zero,
                        "test_y": test_y.astype(np.int64),
                        "pred": selected_pred.astype(np.int64),
                        "pred_attr": state["pred_attr"].astype(np.float32),
                        "z_test": state["z_test"],
                        "feature_test": state["f_test"],
                        "test_x": test_x.astype(np.float32),
                        "graph_attr_unseen": test_attr_m.astype(np.float32),
                        "calibration_mse": state["calibration"]["mse"],
                        "reliability_weights": state["calibration"]["weights"],
                        "calibration_pred_val": state["calibration"]["pred_val"],
                        "calibration_true_val": state["calibration"]["true_val"],
                    }
                )
            rows_by_mode[weighting].append(
                {
                    "task": task_idx + 1,
                    "unseen": unseen,
                    "accuracy": state["acc"],
                    "per_fault": per_fault,
                    "selected_mode": state["mode"],
                    "prediction_diag": state["diag"],
                    "selected_epoch": state["epoch"],
                    "checkpoint_selection": args.checkpoint_selection,
                    "selection_score": state["score"],
                    "attr_weighting": weighting,
                }
            )

    summaries = {}
    for weighting in weighting_modes:
        out_dir = reliability_output_dir(args.out_dir, weighting, args.compare_reliability)
        out_dir.mkdir(parents=True, exist_ok=True)
        if args.save_artifacts:
            for item in artifacts_by_mode[weighting]:
                task_num = item["task"]
                payload = {key: value for key, value in item.items() if key != "task"}
                np.savez_compressed(out_dir / f"gsan_artifacts_task{task_num}.npz", **payload)
        details = pd.DataFrame(rows_by_mode[weighting])
        details.to_csv(out_dir / "gsan_details.csv", index=False)
        np.save(out_dir / "graph_enhanced_attributes.npy", graph_attr)
        np.save(out_dir / "class_graph_adj.npy", adj)
        if refiner_metadata is not None:
            np.save(out_dir / "refiner_output.npy", refined_output)
            if learned_edges is not None:
                np.savez_compressed(out_dir / "learned_edge_state.npz", **learned_edges)
                np.save(out_dir / "initial_class_graph_adj.npy", initial_adj)
            else:
                torch.save(refiner.state_dict(), out_dir / "refiner_state.pt")
            (out_dir / "refiner_metadata.json").write_text(json.dumps(refiner_metadata, indent=2), encoding="utf-8")
        config = dict(vars(args))
        config["attr_weighting"] = weighting
        summary = {
            "method": "GSAN_graph_enhanced_attributes",
            "mean_accuracy": float(details["accuracy"].mean()),
            "task_accuracy": details["accuracy"].tolist(),
            "config": config,
        }
        if refiner_metadata is not None:
            summary["method"] = f"GSAN_{prototype_refiner}_prototype_replacement"
            summary["training_config"] = cfg
        (out_dir / "gsan_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        summaries[weighting] = summary
        print(json.dumps(summary, indent=2))
    return summaries


def main():
    parser = argparse.ArgumentParser(description="Graph Semantic Attribute Network for TEP ZSL.")
    parser.add_argument("--semantic-path", default="data/semantic_cache/tep_fault_semantics_qwen3_embedding.npy")
    parser.add_argument("--attr-path", default="data/TEP_attribute_matrix.npy")
    parser.add_argument("--data-path", default="./TEP_data/*.dat")
    parser.add_argument("--out-dir", default="paper_workspace/gsan_tep")
    parser.add_argument("--seed", type=int, default=904)
    parser.add_argument("--epochs", type=int, default=900)
    parser.add_argument(
        "--tasks",
        type=int,
        nargs="*",
        default=None,
        help="Optional 1-based task IDs to run while preserving the original task numbering.",
    )
    parser.add_argument(
        "--task-epochs",
        type=int,
        nargs="*",
        default=None,
        help="Optional per-task epoch counts, e.g. epochs selected by seen-validation and then used for full seen retraining.",
    )
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--lambda-attr", type=float, default=None)
    parser.add_argument("--lambda-reconstruction", type=float, default=None)
    parser.add_argument("--lambda-compatibility", type=float, default=None)
    parser.add_argument("--lambda-off-diagonal", type=float, default=None)
    parser.add_argument("--center-coef", type=float, default=None)
    parser.add_argument("--graph-mode", choices=["attribute", "semantic", "hybrid"], default="hybrid")
    parser.add_argument("--prototype-refiner", choices=["fixed", "gcn", "gat", "learned_adjacency"], default="fixed",
                        help="Offline class-prototype preparation used consistently before signal training and during matching.")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--beta", type=float, default=0.01)
    parser.add_argument("--rho", type=float, default=0.5, help="Hybrid graph balance. Higher values trust engineering attributes more than LLM semantics.")
    parser.add_argument("--adaptive-betas", type=float, nargs="*", default=None)
    parser.add_argument("--residual-dim", type=int, default=0)
    parser.add_argument("--include-original", action="store_true")
    parser.add_argument("--attr-predictor", choices=["nb", "nb_binary", "nb_prob", "ridge"], default="nb_binary")
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--attr-weighting", choices=["uniform", "inverse_mse", "binary_accuracy"], default="uniform")
    parser.add_argument(
        "--compare-reliability",
        action="store_true",
        help="Evaluate uniform and inverse-MSE attribute matching on the same trained model trajectory.",
    )
    parser.add_argument("--attr-weight-val-ratio", type=float, default=0.2)
    parser.add_argument("--attr-weight-eps", type=float, default=1e-3)
    parser.add_argument("--attr-weight-clip", type=float, default=5.0)
    parser.add_argument(
        "--checkpoint-selection",
        choices=["best_test", "final", "seen_val"],
        default="best_test",
        help="best_test reproduces the development setting; final reports the last evaluated epoch; seen_val selects by held-out seen-class accuracy.",
    )
    parser.add_argument("--seen-val-ratio", type=float, default=0.2)
    parser.add_argument("--balanced-assignment", action="store_true")
    parser.add_argument("--save-artifacts", action="store_true", help="Save task-level predictions and latent features for paper figures.")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
