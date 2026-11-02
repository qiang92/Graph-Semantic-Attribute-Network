import argparse
import time
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sce_tep import CONFIG, TEST_TASKS, CenterLossModule
from utils.read_tep_data import create_tep_data, load_tep_data
from utils.run_gsan_tep import (
    AttrDataset,
    DynamicGraphAttributeNet,
    graph_enhance_attributes,
    learn_attr_calibration_seen,
    match_attr,
    predict_attr,
)
from utils.run_llm_gcn_zsl import normalize_rows, set_seed


def per_class_accuracy(y_true, y_pred, labels):
    values = []
    for label in labels:
        mask = y_true == label
        values.append(float((y_pred[mask] == label).mean()))
    return float(np.mean(values))


def harmonic(seen, unseen):
    return 0.0 if seen + unseen == 0 else 2.0 * seen * unseen / (seen + unseen)


def calibrated_predict(pred_attr, prototypes, labels, weights, penalized_labels, gamma):
    distance = (
        (pred_attr[:, None, :] - prototypes[None, :, :]) ** 2
        * weights[None, None, :]
    ).sum(axis=2)
    penalty_mask = np.isin(labels, penalized_labels)
    distance[:, penalty_mask] += gamma
    return labels[np.argmin(distance, axis=1)]


def select_seen_only_gamma(features, attributes, labels, prototypes, weights, seed):
    unique = np.unique(labels)
    folds = np.array_split(np.random.default_rng(seed).permutation(unique), 3)
    gamma_grid = np.linspace(0.0, 10.0, 41)
    scores = np.zeros(gamma_grid.shape[0], dtype=np.float64)
    fold_scores = []
    for fold_index, pseudo_unseen in enumerate(folds):
        current_fold_scores = []
        pseudo_seen = np.setdiff1d(unique, pseudo_unseen)
        train_pool = np.flatnonzero(np.isin(labels, pseudo_seen))
        train_index, seen_eval_index = train_test_split(
            train_pool,
            test_size=0.2,
            random_state=seed + fold_index,
            stratify=labels[train_pool],
        )
        unseen_eval_index = np.flatnonzero(np.isin(labels, pseudo_unseen))
        evaluation_index = np.concatenate([seen_eval_index, unseen_eval_index])
        prediction = predict_attr(
            features[train_index],
            attributes[train_index],
            features[evaluation_index],
            "nb_binary",
            1.0,
        )
        y_eval = labels[evaluation_index]
        for gamma_index, gamma in enumerate(gamma_grid):
            y_pred = calibrated_predict(
                prediction,
                prototypes[unique],
                unique,
                weights,
                pseudo_seen,
                gamma,
            )
            seen_acc = per_class_accuracy(y_eval, y_pred, pseudo_seen)
            unseen_acc = per_class_accuracy(y_eval, y_pred, pseudo_unseen)
            fold_h = harmonic(seen_acc, unseen_acc)
            scores[gamma_index] += fold_h
            current_fold_scores.append(fold_h)
        fold_scores.append(current_fold_scores)
        np.savez_compressed(select_seen_only_gamma.evidence_dir / f"gamma_fold_{fold_index}.npz",
            pseudo_seen=pseudo_seen, pseudo_unseen=pseudo_unseen, train_index=train_index,
            seen_eval_index=seen_eval_index, unseen_eval_index=unseen_eval_index,
            evaluation_index=evaluation_index, y_eval=y_eval, prediction=prediction)
    scores /= len(folds)
    np.savez_compressed(select_seen_only_gamma.evidence_dir / "gamma_curve.npz",
        gamma_grid=gamma_grid, mean_h=scores, fold_h=np.asarray(fold_scores))
    best = int(np.argmax(scores))
    return float(gamma_grid[best]), float(scores[best] * 100.0)


def run_seed(args, seed):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = dict(CONFIG)
    cfg["lambda_"] = list(CONFIG["lambda_"])
    cfg["epochs"] = args.epochs
    attr = np.load(args.attr_path).astype(np.float32)
    semantic = normalize_rows(np.load(args.semantic_path).astype(np.float32))
    graph_attr, _ = graph_enhance_attributes(
        attr,
        semantic,
        "hybrid",
        args.topk,
        args.beta,
        args.rho,
        0,
        False,
    )
    data_list = load_tep_data(args.data_path, sma_window=cfg.get("sma_window", 3))
    rows = []
    all_labels = np.arange(attr.shape[0], dtype=np.int64)

    for task_index, task in enumerate(TEST_TASKS, start=1):
        task_started = time.perf_counter()
        task_dir = Path(args.out_dir) / f"seed_{seed}" / f"task_{task_index}"
        task_dir.mkdir(parents=True, exist_ok=True)
        print(f"START seed={seed} task={task_index}", flush=True)
        train_x, train_y, train_attr, _, unseen_x, unseen_y, _, _, _ = (
            create_tep_data(
                data_list, graph_attr, task["seen_classes"], task["test_class"]
            )
        )
        fit_index, seen_test_index = train_test_split(
            np.arange(train_x.shape[0]),
            test_size=args.seen_test_ratio,
            random_state=seed + task_index,
            stratify=train_y,
        )
        fit_x = train_x[fit_index]
        fit_y = train_y[fit_index]
        fit_attr = train_attr[fit_index]
        seen_test_x = train_x[seen_test_index]
        seen_test_y = train_y[seen_test_index]

        fit_x_t = torch.as_tensor(fit_x, dtype=torch.float32, device=device)
        fit_attr_t = torch.as_tensor(fit_attr, dtype=torch.float32, device=device)
        fit_y_t = torch.as_tensor(fit_y, dtype=torch.long, device=device)
        seen_test_t = torch.as_tensor(
            seen_test_x, dtype=torch.float32, device=device
        )
        unseen_t = torch.as_tensor(unseen_x, dtype=torch.float32, device=device)
        unique_labels = np.unique(fit_y)
        label_to_local = {label: index for index, label in enumerate(unique_labels)}
        loader = DataLoader(
            AttrDataset(fit_x_t, fit_attr_t, fit_y_t),
            batch_size=cfg["batch_size"],
            shuffle=True,
        )
        model = DynamicGraphAttributeNet(
            input_dim=fit_x.shape[1],
            attr_dim=graph_attr.shape[1],
            hidden=cfg["dim"][0],
            latent=cfg["dim"][1],
            lambda_=cfg["lambda_"],
            dropout=cfg["dropout"],
        ).to(device)
        center = CenterLossModule(len(unique_labels), cfg["dim"][1]).to(device)
        optimizer = torch.optim.RMSprop(
            list(model.parameters()) + list(center.parameters()), lr=cfg["lr"]
        )
        scheduler = MultiStepLR(optimizer, milestones=[450], gamma=0.5)
        for epoch in range(cfg["epochs"]):
            model.train()
            for batch_x, batch_attr, batch_y in loader:
                optimizer.zero_grad()
                z_x, z_a = model.transform(batch_x, batch_attr)
                rec_x, rec_a = model.reconstruction(z_x, z_a)
                local_y = torch.as_tensor(
                    [label_to_local[int(value)] for value in batch_y.cpu().numpy()],
                    dtype=torch.long,
                    device=device,
                )
                loss = (
                    cfg["lambda_"][0] * F.mse_loss(batch_attr, rec_a)
                    + cfg["lambda_"][1] * F.mse_loss(batch_x, rec_x)
                    + cfg["lambda_"][2] * model.compability_loss(z_x, z_a)
                    + cfg["center_coef"] * center(z_x, local_y)
                )
                loss.backward()
                optimizer.step()
            scheduler.step()
            if (epoch + 1) % 100 == 0:
                print(f"PROGRESS seed={seed} task={task_index} epoch={epoch+1} elapsed={time.perf_counter()-task_started:.1f}", flush=True)

        model.eval()
        with torch.no_grad():
            z_fit, _ = model.transform(fit_x_t, fit_attr_t)
            zero_seen = torch.zeros(
                seen_test_t.shape[0], graph_attr.shape[1], device=device
            )
            zero_unseen = torch.zeros(
                unseen_t.shape[0], graph_attr.shape[1], device=device
            )
            z_seen, _ = model.transform(seen_test_t, zero_seen)
            z_unseen, _ = model.transform(unseen_t, zero_unseen)
            f_fit = torch.cat([z_fit, fit_x_t], dim=1).cpu().numpy()
            f_seen = torch.cat([z_seen, seen_test_t], dim=1).cpu().numpy()
            f_unseen = torch.cat([z_unseen, unseen_t], dim=1).cpu().numpy()
        pred_seen_attr = predict_attr(
            f_fit, fit_attr, f_seen, args.attr_predictor, 1.0
        )
        pred_unseen_attr = predict_attr(
            f_fit, fit_attr, f_unseen, args.attr_predictor, 1.0
        )
        calibration_args = argparse.Namespace(
            attr_weighting="inverse_mse",
            attr_weight_val_ratio=args.calibration_ratio,
            seed=seed,
            attr_predictor=args.attr_predictor,
            ridge_alpha=1.0,
            attr_weight_eps=1e-3,
            attr_weight_clip=2.0,
        )
        weights = learn_attr_calibration_seen(
            f_fit, fit_y, fit_attr, calibration_args
        )["weights"]
        select_seen_only_gamma.evidence_dir = task_dir
        gamma, pseudo_h = select_seen_only_gamma(
            f_fit, fit_attr, fit_y, graph_attr, weights, seed + task_index
        )
        np.savez_compressed(task_dir / "features.npz", f_fit=f_fit, f_seen=f_seen,
            f_unseen=f_unseen, fit_y=fit_y, fit_attr=fit_attr, fit_index=fit_index,
            seen_test_index=seen_test_index, seen_test_y=seen_test_y, unseen_y=unseen_y,
            graph_attr=graph_attr, weights=weights, pred_seen_attr=pred_seen_attr,
            pred_unseen_attr=pred_unseen_attr, gamma=gamma, pseudo_h=pseudo_h)
        torch.save({"model": model.state_dict(), "config": cfg, "seed": seed,
            "task": task_index, "epoch": cfg["epochs"]}, task_dir / "final_model.pt")
        seen_labels = np.unique(seen_test_y)
        unseen_labels = np.unique(unseen_y)
        for method, penalty in (("Uncalibrated", 0.0), ("Seen-only calibrated", gamma)):
            pred_seen = calibrated_predict(
                pred_seen_attr, graph_attr, all_labels, weights, seen_labels, penalty
            )
            pred_unseen = calibrated_predict(
                pred_unseen_attr, graph_attr, all_labels, weights, seen_labels, penalty
            )
            seen_acc = per_class_accuracy(seen_test_y, pred_seen, seen_labels)
            unseen_acc = per_class_accuracy(unseen_y, pred_unseen, unseen_labels)
            rows.append(
                {
                    "seed": seed,
                    "task": task_index,
                    "method": method,
                    "gamma": penalty,
                    "pseudo_unseen_h": pseudo_h,
                    "seen_accuracy": seen_acc * 100.0,
                    "unseen_accuracy": unseen_acc * 100.0,
                    "harmonic_mean": harmonic(seen_acc, unseen_acc) * 100.0,
                    "seen_test_samples": int(seen_test_y.size),
                    "unseen_test_samples": int(unseen_y.size),
                }
            )
            method_key = "uncalibrated" if penalty == 0 and method == "Uncalibrated" else "calibrated"
            np.savez_compressed(task_dir / f"{method_key}_predictions.npz",
                seen=pred_seen, unseen=pred_unseen)
            print(rows[-1], flush=True)
        pd.DataFrame(rows[-2:]).to_csv(task_dir / "results.csv", index=False)
        (task_dir / "timing.json").write_text(json.dumps({"seed": seed, "task": task_index,
            "seconds": time.perf_counter()-task_started}), encoding="utf-8")
    return rows


def main():
    parser = argparse.ArgumentParser(description="TEP generalized ZSFD evaluation.")
    parser.add_argument("--seed", type=int, action="append", required=True)
    parser.add_argument("--epochs", type=int, default=900)
    parser.add_argument("--seen-test-ratio", type=float, default=0.2)
    parser.add_argument("--calibration-ratio", type=float, default=0.2)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--beta", type=float, default=0.02)
    parser.add_argument("--rho", type=float, default=0.5)
    parser.add_argument("--attr-predictor", default="nb_binary")
    parser.add_argument("--attr-path", default="data/TEP_attribute_matrix.npy")
    parser.add_argument(
        "--semantic-path",
        default="data/semantic_cache/tep_fault_semantics_qwen3_embedding.npy",
    )
    parser.add_argument("--data-path", default="./TEP_data/*.dat")
    parser.add_argument(
        "--out-dir", default="major_revision/experiments/gzsfd"
    )
    args = parser.parse_args()

    rows = []
    for seed in args.seed:
        rows.extend(run_seed(args, seed))
    detail = pd.DataFrame(rows)
    per_run = (
        detail.groupby(["seed", "method"], as_index=False)
        .agg(
            seen_accuracy=("seen_accuracy", "mean"),
            unseen_accuracy=("unseen_accuracy", "mean"),
            harmonic_mean=("harmonic_mean", "mean"),
        )
    )
    aggregate = (
        per_run.groupby("method", as_index=False)
        .agg(
            runs=("seed", "count"),
            seen_accuracy_mean=("seen_accuracy", "mean"),
            seen_accuracy_std=("seen_accuracy", "std"),
            unseen_accuracy_mean=("unseen_accuracy", "mean"),
            unseen_accuracy_std=("unseen_accuracy", "std"),
            harmonic_mean=("harmonic_mean", "mean"),
            harmonic_std=("harmonic_mean", "std"),
        )
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(out_dir / "gzsfd_task_results.csv", index=False)
    per_run.to_csv(out_dir / "gzsfd_run_results.csv", index=False)
    aggregate.to_csv(out_dir / "gzsfd_summary.csv", index=False)
    (out_dir / "gzsfd_protocol.json").write_text(
        json.dumps(
            {
                "seen_holdout_ratio": args.seen_test_ratio,
                "candidate_prototypes": "all 15 TEP fault classes",
                "calibration": "gamma selected using three pseudo-unseen folds formed only from seen classes",
                "unseen_samples_used_for_training_or_calibration": False,
                "seeds": args.seed,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(aggregate.to_string(index=False))


if __name__ == "__main__":
    main()
