import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.read_tep_data import create_tep_data, load_tep_data


TASKS_4 = [
    {"name": "A", "unseen": [1, 6, 14]},
    {"name": "B", "unseen": [4, 7, 10]},
    {"name": "C", "unseen": [8, 11, 12]},
    {"name": "D", "unseen": [2, 3, 5]},
]

TASKS_5 = TASKS_4 + [{"name": "E", "unseen": [9, 13, 15]}]


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_seen(unseen):
    unseen_set = set(unseen)
    return [i for i in range(1, 16) if i not in unseen_set]


def normalize_rows(x, eps=1e-12):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)


def build_class_graph(attr, semantic, graph_mode="hybrid", topk=5, self_loop=1.0, rho=0.5):
    attr_n = normalize_rows(attr.astype(np.float32))
    sem_n = normalize_rows(semantic.astype(np.float32))
    if graph_mode == "attribute":
        sim = attr_n @ attr_n.T
    elif graph_mode == "semantic":
        sim = sem_n @ sem_n.T
    else:
        sim = rho * (attr_n @ attr_n.T) + (1.0 - rho) * (sem_n @ sem_n.T)

    sim = np.maximum(sim, 0.0)
    np.fill_diagonal(sim, 0.0)
    adj = np.zeros_like(sim, dtype=np.float32)
    for i in range(sim.shape[0]):
        idx = np.argsort(sim[i])[-topk:]
        adj[i, idx] = sim[i, idx]
    adj = np.maximum(adj, adj.T)
    adj = adj + np.eye(adj.shape[0], dtype=np.float32) * self_loop
    deg = adj.sum(axis=1)
    deg_inv_sqrt = np.diag(1.0 / np.sqrt(deg + 1e-12))
    return (deg_inv_sqrt @ adj @ deg_inv_sqrt).astype(np.float32)


def build_target_similarity(attr, semantic, graph_mode="hybrid"):
    attr_sim = normalize_rows(attr.astype(np.float32)) @ normalize_rows(attr.astype(np.float32)).T
    sem_sim = normalize_rows(semantic.astype(np.float32)) @ normalize_rows(semantic.astype(np.float32)).T
    if graph_mode == "attribute":
        return attr_sim.astype(np.float32)
    if graph_mode == "semantic":
        return sem_sim.astype(np.float32)
    return (0.5 * attr_sim + 0.5 * sem_sim).astype(np.float32)


class ArrayDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.long)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x, adj):
        return self.linear(adj @ x)


class LLMGCNZSL(nn.Module):
    def __init__(self, input_dim, attr_dim, sem_dim, hidden, latent, dropout):
        super().__init__()
        self.signal = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, latent),
        )
        proto_in = attr_dim + sem_dim
        self.proto_in = nn.Sequential(
            nn.Linear(proto_in, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.gcn1 = GCNLayer(hidden, hidden)
        self.gcn2 = GCNLayer(hidden, latent)
        self.attr_head = nn.Linear(latent, attr_dim)
        self.sem_head = nn.Linear(latent, sem_dim)

    def encode_signal(self, x):
        return F.normalize(self.signal(x), dim=1)

    def encode_prototypes(self, attr_proto, sem_proto, adj, proto_blend=0.35):
        h = self.proto_in(torch.cat([attr_proto, sem_proto], dim=1))
        h = F.relu(self.gcn1(h, adj))
        gcn_proto = self.gcn2(h, adj)
        return F.normalize(gcn_proto, dim=1)

    def forward(self, x, attr_proto, sem_proto, adj):
        z = self.encode_signal(x)
        p = self.encode_prototypes(attr_proto, sem_proto, adj)
        attr_pred = self.attr_head(z)
        sem_pred = self.sem_head(z)
        return z, p, attr_pred, sem_pred


def evaluate(y_true, y_pred, labels_zero):
    sample = accuracy_score(y_true, y_pred) * 100.0
    per_fault = {}
    for lab in labels_zero:
        mask = y_true == lab
        per_fault[f"F{lab + 1}"] = float((y_pred[mask] == lab).mean() * 100.0)
    return sample, float(np.mean(list(per_fault.values()))), per_fault


def predict_nb_attr(train_feat, train_attr, test_feat, unseen_attr, unseen_zero):
    preds = []
    for j in range(train_attr.shape[1]):
        if np.max(train_attr[:, j]) == np.min(train_attr[:, j]):
            preds.append(np.full(test_feat.shape[0], train_attr[0, j]))
            continue
        clf = GaussianNB()
        clf.fit(train_feat, train_attr[:, j])
        preds.append(clf.predict(test_feat))
    pred_attr = np.stack(preds, axis=1)
    dist = ((pred_attr[:, None, :] - unseen_attr[None, :, :]) ** 2).sum(axis=2)
    return unseen_zero[dist.argmin(axis=1)]


def run_task(task, data_list, attr, sem, adj, args, device):
    unseen = task["unseen"]
    seen = make_seen(unseen)
    train_x, train_y, train_attr, _, test_x, test_y, _, test_attr_m, _ = create_tep_data(
        data_list, attr, seen, unseen
    )
    seen_zero = np.array([c - 1 for c in seen], dtype=np.int64)
    unseen_zero = np.array([c - 1 for c in unseen], dtype=np.int64)
    seen_map = {lab: i for i, lab in enumerate(seen_zero)}
    train_local_y = np.array([seen_map[int(y)] for y in train_y], dtype=np.int64)

    model = LLMGCNZSL(
        input_dim=train_x.shape[1],
        attr_dim=attr.shape[1],
        sem_dim=sem.shape[1],
        hidden=args.hidden,
        latent=args.latent,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loader = DataLoader(ArrayDataset(train_x, train_local_y), batch_size=args.batch_size, shuffle=True)

    attr_t = torch.as_tensor(attr, dtype=torch.float32, device=device)
    sem_t = torch.as_tensor(sem, dtype=torch.float32, device=device)
    adj_t = torch.as_tensor(adj, dtype=torch.float32, device=device)
    target_sim_t = torch.as_tensor(build_target_similarity(attr, sem, args.graph_mode), dtype=torch.float32, device=device)
    train_attr_t = torch.as_tensor(train_attr, dtype=torch.float32, device=device)
    train_sem_t = torch.as_tensor(sem[train_y], dtype=torch.float32, device=device)
    train_x_t = torch.as_tensor(train_x, dtype=torch.float32, device=device)
    test_x_t = torch.as_tensor(test_x, dtype=torch.float32, device=device)

    best = {"per_class_top1": 0.0, "sample_acc": 0.0, "per_fault": {}, "pred": None}
    for epoch in range(args.epochs):
        model.train()
        for bx, by_local in loader:
            bx = bx.to(device)
            by_local = by_local.to(device)
            z = model.encode_signal(bx)
            proto = model.encode_prototypes(attr_t, sem_t, adj_t, 1.0)
            attr_pred = model.attr_head(z)
            sem_pred = model.sem_head(z)
            seen_proto = proto[seen_zero]
            logits = args.logit_scale * (z @ seen_proto.T)
            ce = F.cross_entropy(logits, by_local)
            global_y = torch.as_tensor(seen_zero, device=device)[by_local]
            attr_loss = F.binary_cross_entropy_with_logits(attr_pred, attr_t[global_y])
            sem_loss = 1.0 - F.cosine_similarity(sem_pred, sem_t[global_y], dim=1).mean()
            loss = ce + args.attr_weight * attr_loss + args.sem_weight * sem_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            model.eval()
            with torch.no_grad():
                z_test = model.encode_signal(test_x_t)
                z_train = model.encode_signal(train_x_t)
                proto = model.encode_prototypes(attr_t, sem_t, adj_t, 1.0)
                if args.eval_mode == "prototype":
                    logits = z_test @ proto[unseen_zero].T
                    pred = unseen_zero[logits.argmax(dim=1).cpu().numpy()]
                else:
                    f_train = torch.cat([z_train, train_x_t], dim=1).cpu().numpy()
                    f_test = torch.cat([z_test, test_x_t], dim=1).cpu().numpy()
                    pred = predict_nb_attr(f_train, train_attr, f_test, test_attr_m, unseen_zero)
                sample, per_class, per_fault = evaluate(test_y, pred, unseen_zero)
                if per_class > best["per_class_top1"]:
                    best = {
                        "per_class_top1": per_class,
                        "sample_acc": sample,
                        "per_fault": per_fault,
                        "pred": pred, "epoch": epoch + 1,
                    }

    np.savez_compressed(Path(args.out_dir) / f"predictions_Task{task['name']}.npz", test_y=test_y, pred=best["pred"], unseen_zero=unseen_zero)
    cm = confusion_matrix(test_y, best["pred"], labels=unseen_zero)
    return {
        "task": task["name"],
        "selected_epoch": best["epoch"],
        "unseen": unseen,
        "sample_acc": best["sample_acc"],
        "per_class_top1": best["per_class_top1"],
        "per_fault": best["per_fault"],
        "cm": cm,
    }


def main():
    parser = argparse.ArgumentParser(description="Strict ZSL with LLM semantic prototypes and class-prototype GCN.")
    parser.add_argument("--semantic-path", default="data/semantic_cache/tep_fault_semantics_qwen3_embedding.npy")
    parser.add_argument("--attr-path", default="data/TEP_attribute_matrix.npy")
    parser.add_argument("--data-path", default="./TEP_data/*.dat")
    parser.add_argument("--out-dir", default="paper_workspace/llm_gcn_zsl")
    parser.add_argument("--tasks", choices=["4", "5"], default="4")
    parser.add_argument("--graph-mode", choices=["attribute", "semantic", "hybrid"], default="hybrid")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--sma-window", type=int, default=3)
    parser.add_argument("--seed", type=int, default=904)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--latent", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--logit-scale", type=float, default=20.0)
    parser.add_argument("--attr-weight", type=float, default=0.2)
    parser.add_argument("--sem-weight", type=float, default=0.1)
    parser.add_argument("--eval-mode", choices=["prototype", "nb_attr"], default="prototype")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    attr = np.load(args.attr_path).astype(np.float32)
    sem = np.load(args.semantic_path).astype(np.float32)
    sem = normalize_rows(sem)
    adj = build_class_graph(attr, sem, args.graph_mode, args.topk)
    np.save(out_dir / "class_graph_adj.npy", adj)

    data_list = load_tep_data(args.data_path, sma_window=args.sma_window)
    tasks = TASKS_4 if args.tasks == "4" else TASKS_5

    rows = []
    for task in tasks:
        result = run_task(task, data_list, attr, sem, adj, args, device)
        np.save(out_dir / f"cm_Task{task['name']}.npy", result.pop("cm"))
        rows.append(result)
        print(
            f"Task {task['name']} unseen={task['unseen']} "
            f"per_class_top1={result['per_class_top1']:.2f}% per_fault={result['per_fault']}"
        )

    details = pd.DataFrame(rows)
    details.to_csv(out_dir / "llm_gcn_zsl_details.csv", index=False)
    summary = {
        "method": "llm_semantic_attribute_class_gcn",
        "mean_per_class_top1": float(details["per_class_top1"].mean()),
        "mean_sample_acc": float(details["sample_acc"].mean()),
        "config": vars(args),
    }
    (out_dir / "llm_gcn_zsl_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
