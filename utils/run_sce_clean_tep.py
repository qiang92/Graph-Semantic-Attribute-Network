import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sce_tep import CONFIG, TEST_TASKS, CenterLossModule, off_diagonal
from utils.read_tep_data import create_tep_data, load_tep_data
from utils.run_llm_gcn_zsl import set_seed


class AttrDataset(Dataset):
    def __init__(self, x, a, y):
        self.x = x
        self.a = a
        self.y = y

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.a[idx], self.y[idx]


class EmbeddingNet(nn.Module):
    def __init__(self, dim, lambda_, dropout):
        super().__init__()
        self.l11 = nn.Linear(dim[0], dim[0])
        self.l12 = nn.Linear(dim[0], dim[1])
        self.l13 = nn.Linear(2 * dim[1], 52)
        self.l21 = nn.Linear(20, dim[0])
        self.l22 = nn.Linear(dim[0], dim[1])
        self.l23 = nn.Linear(2 * dim[1], 20)
        self.bn1 = nn.BatchNorm1d(dim[0])
        self.ln1 = nn.LayerNorm(dim[0])
        self.bn2 = nn.BatchNorm1d(dim[1])
        self.dropout = nn.Dropout(dropout)
        self.lambda_ = lambda_

    def transform(self, x, a):
        h1 = self.bn1(self.l11(x))
        z1 = self.l12(self.dropout(F.relu(self.ln1(h1))))
        h2 = self.bn1(self.l21(a))
        z2 = self.l22(self.dropout(F.relu(self.ln1(h2))))
        return z1, z2

    def reconstruction(self, z1, z2):
        x_rec = self.l13(torch.cat([z1, z2], dim=1))
        a_rec = torch.sigmoid(self.l23(torch.cat([z2, z1], dim=1)))
        return x_rec, a_rec

    def compatibility_loss(self, z1, z2):
        n = z1.shape[0]
        c = self.bn2(z1).T @ self.bn2(z2) / n
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        return on_diag + self.lambda_[3] * off_diag


def fit_predict_nb(train_feat, train_attr, test_feat):
    preds = []
    for j in range(train_attr.shape[1]):
        col = train_attr[:, j]
        if np.max(col) == np.min(col):
            preds.append(np.full(test_feat.shape[0], float(col[0])))
            continue
        clf = GaussianNB()
        clf.fit(train_feat, col)
        preds.append(clf.predict(test_feat))
    return np.stack(preds, axis=1)


def match_attrs(pred_attr, prototypes, labels):
    dist = ((pred_attr[:, None, :] - prototypes[None, :, :]) ** 2).sum(axis=2)
    return labels[dist.argmin(axis=1)]


def run(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = dict(CONFIG)
    cfg["epochs"] = args.epochs
    cfg["sma_window"] = args.sma_window
    attr_all = np.load(args.attr_path).astype(np.float32)
    data_list = load_tep_data(args.data_path, sma_window=args.sma_window)
    rows = []
    artifacts = []

    for task_idx, task in enumerate(TEST_TASKS):
        unseen = task["test_class"]
        seen = task["seen_classes"]
        unseen_zero = np.array([c - 1 for c in unseen], dtype=np.int64)
        print(f"--- Task {task_idx + 1}: unseen={unseen} ---")
        train_x, train_y, train_attr, _, test_x, test_y, _, test_attr_m, _ = create_tep_data(
            data_list, attr_all, seen, unseen
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
        test_x_t = torch.as_tensor(test_x, dtype=torch.float32, device=device)
        if val_idx is not None:
            val_x_t = torch.as_tensor(train_x[val_idx], dtype=torch.float32, device=device)
        labels = np.unique(train_y_fit)
        label_to_idx = {lab: i for i, lab in enumerate(labels)}
        loader = DataLoader(AttrDataset(train_x_t, train_attr_t, train_y_t), batch_size=cfg["batch_size"], shuffle=True)

        model = EmbeddingNet(cfg["dim"], cfg["lambda_"], cfg["dropout"]).to(device)
        center = CenterLossModule(len(labels), cfg["dim"][1]).to(device)
        optimizer = torch.optim.RMSprop(list(model.parameters()) + list(center.parameters()), lr=cfg["lr"])
        scheduler = MultiStepLR(optimizer, milestones=[450], gamma=0.5)

        selected_acc = None
        selected_score = None
        selected_pred = None
        selected_pred_attr = None
        selected_f_train = None
        selected_train_attr = None
        selected_train_y = None
        selected_z_test = None
        selected_f_test = None
        selected_epoch = None

        for epoch in range(cfg["epochs"]):
            model.train()
            for bx, ba, by in loader:
                optimizer.zero_grad()
                z1, z2 = model.transform(bx, ba)
                x_rec, a_rec = model.reconstruction(z1, z2)
                local = torch.as_tensor([label_to_idx[int(y)] for y in by.detach().cpu().numpy()], dtype=torch.long, device=device)
                loss = (
                    cfg["lambda_"][0] * F.mse_loss(ba, a_rec)
                    + cfg["lambda_"][1] * F.mse_loss(bx, x_rec)
                    + cfg["lambda_"][2] * model.compatibility_loss(z1, z2)
                    + cfg["center_coef"] * center(z1, local)
                )
                loss.backward()
                optimizer.step()
            scheduler.step()

            if (epoch + 1) % args.eval_every == 0 or epoch == cfg["epochs"] - 1:
                model.eval()
                with torch.no_grad():
                    z_train, _ = model.transform(train_x_t, train_attr_t)
                    z_test, _ = model.transform(
                        test_x_t,
                        torch.zeros(test_x_t.size(0), attr_all.shape[1], dtype=torch.float32, device=device),
                    )
                    f_train = torch.cat([z_train, train_x_t], dim=1).cpu().numpy()
                    f_test = torch.cat([z_test, test_x_t], dim=1).cpu().numpy()
                    if val_idx is not None:
                        z_val, _ = model.transform(
                            val_x_t,
                            torch.zeros(val_x_t.size(0), attr_all.shape[1], dtype=torch.float32, device=device),
                        )
                        f_val = torch.cat([z_val, val_x_t], dim=1).cpu().numpy()
                pred_attr = fit_predict_nb(f_train, train_attr_fit, f_test)
                pred = match_attrs(pred_attr, test_attr_m, unseen_zero)
                acc = accuracy_score(test_y, pred)
                score = acc
                if args.checkpoint_selection == "seen_val":
                    val_pred_attr = fit_predict_nb(f_train, train_attr_fit, f_val)
                    seen_labels = np.unique(train_y)
                    val_pred = match_attrs(val_pred_attr, attr_all[seen_labels], seen_labels)
                    score = accuracy_score(train_y[val_idx], val_pred)
                if args.checkpoint_selection == "best_test":
                    should_select = selected_acc is None or acc > selected_acc
                elif args.checkpoint_selection == "final":
                    should_select = True
                elif args.checkpoint_selection == "seen_val":
                    should_select = selected_score is None or score > selected_score
                else:
                    raise ValueError(args.checkpoint_selection)
                if should_select:
                    selected_acc = float(acc)
                    selected_score = float(score)
                    selected_pred = pred
                    selected_pred_attr = pred_attr
                    selected_f_train = f_train.astype(np.float32)
                    selected_train_attr = train_attr_fit.astype(np.float32)
                    selected_train_y = train_y_fit.astype(np.int64)
                    selected_z_test = z_test.detach().cpu().numpy().astype(np.float32)
                    selected_f_test = f_test.astype(np.float32)
                    selected_epoch = epoch + 1

        per_fault = {f"F{lab + 1}": float((selected_pred[test_y == lab] == lab).mean() * 100.0) for lab in unseen_zero}
        print(f"Task {task_idx + 1} acc={selected_acc:.4f} epoch={selected_epoch} per_fault={per_fault}")
        if args.save_artifacts:
            artifacts.append(
                {
                    "task": task_idx + 1,
                    "unseen_zero": unseen_zero,
                    "test_y": test_y.astype(np.int64),
                    "pred": selected_pred.astype(np.int64),
                    "pred_attr": selected_pred_attr.astype(np.float32),
                    "train_feature": selected_f_train,
                    "train_attr": selected_train_attr,
                    "train_y": selected_train_y,
                    "z_test": selected_z_test,
                    "feature_test": selected_f_test,
                    "test_x": test_x.astype(np.float32),
                }
            )
        rows.append({
            "task": task_idx + 1,
            "unseen": unseen,
            "accuracy": selected_acc,
            "selected_epoch": selected_epoch,
            "selection_score": selected_score,
            "per_fault": per_fault,
        })

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.save_artifacts:
        for item in artifacts:
            task_num = item.pop("task")
            np.savez_compressed(out_dir / f"sce_artifacts_task{task_num}.npz", **item)
    details = pd.DataFrame(rows)
    details.to_csv(out_dir / "sce_clean_details.csv", index=False)
    summary = {
        "method": "SCE_clean_checkpoint_audit",
        "mean_accuracy": float(details["accuracy"].mean()),
        "task_accuracy": details["accuracy"].tolist(),
        "config": vars(args),
    }
    (out_dir / "sce_clean_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attr-path", default="data/TEP_attribute_matrix.npy")
    parser.add_argument("--data-path", default="./TEP_data/*.dat")
    parser.add_argument("--out-dir", default="paper_workspace/sce_clean")
    parser.add_argument("--seed", type=int, default=904)
    parser.add_argument("--epochs", type=int, default=900)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--sma-window", type=int, default=3)
    parser.add_argument("--checkpoint-selection", choices=["best_test", "final", "seen_val"], default="final")
    parser.add_argument("--seen-val-ratio", type=float, default=0.2)
    parser.add_argument("--save-artifacts", action="store_true", help="Save task-level predictions and latent features for paper figures.")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
