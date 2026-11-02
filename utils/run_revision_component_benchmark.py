import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.run_llm_gcn_zsl import build_class_graph, normalize_rows, set_seed


def predict_nb(train_feature, train_attr, test_feature, probabilistic=False):
    predictions = []
    for index in range(train_attr.shape[1]):
        target = (train_attr[:, index] >= 0.5).astype(np.int64)
        if np.max(target) == np.min(target):
            predictions.append(np.full(test_feature.shape[0], float(target[0])))
            continue
        model = GaussianNB().fit(train_feature, target)
        if probabilistic:
            classes = list(model.classes_)
            predictions.append(model.predict_proba(test_feature)[:, classes.index(1)])
        else:
            predictions.append(model.predict(test_feature))
    return np.stack(predictions, axis=1)


def inverse_mse_weights(mse, eps=1e-3, clip=2.0):
    weights = 1.0 / (np.asarray(mse, dtype=np.float64) + eps)
    weights /= weights.mean() + 1e-12
    return np.clip(weights, 0.0, clip)


def entropy_weights(probabilities, clip=2.0):
    p = np.clip(probabilities, 1e-6, 1.0 - 1e-6)
    entropy = -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p)).mean(axis=0)
    confidence = 1.0 - entropy / np.log(2.0)
    confidence /= confidence.mean() + 1e-12
    return np.clip(confidence, 0.0, clip)


def match(pred_attr, prototypes, labels, weights):
    distance = (
        (pred_attr[:, None, :] - prototypes[None, :, :]) ** 2
        * weights[None, None, :]
    ).sum(axis=2)
    return labels[np.argmin(distance, axis=1)]


def learn_adjacency(attr, seed, steps=800, lr=0.05):
    set_seed(seed)
    x = torch.tensor(attr, dtype=torch.float32)
    logits = nn.Parameter(torch.zeros(x.shape[0], x.shape[0]))
    with torch.no_grad():
        cosine = normalize_rows(attr) @ normalize_rows(attr).T
        logits.copy_(torch.tensor(cosine, dtype=torch.float32))
        logits.fill_diagonal_(-12.0)
    optimizer = torch.optim.Adam([logits], lr=lr)
    mask = torch.eye(x.shape[0], dtype=torch.bool)
    for _ in range(steps):
        optimizer.zero_grad()
        weights = torch.softmax(logits.masked_fill(mask, -1e9), dim=1)
        reconstruction = weights @ x
        loss = F.mse_loss(reconstruction, x) + 1e-3 * weights.pow(2).mean()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        weights = torch.softmax(logits.masked_fill(mask, -1e9), dim=1)
    return weights.cpu().numpy().astype(np.float32)


def learn_hybrid_edge_weights(attr, adjacency, steps=800, lr=0.05):
    """Reconstruct attributes by tuning positive weights on fixed hybrid support."""
    # Raw self-loops are one, so normalized diagonal entries recover degrees.
    degree = 1.0 / np.diag(adjacency)
    raw = adjacency * np.sqrt(degree[:, None] * degree[None, :])
    np.fill_diagonal(raw, 1.0)
    upper = np.triu(raw > 0, k=1)
    row, col = np.where(upper)
    x = torch.tensor(attr, dtype=torch.float32)
    initial = torch.tensor(raw[row, col], dtype=torch.float32)
    logits = nn.Parameter(torch.log(torch.expm1(initial)))
    optimizer = torch.optim.Adam([logits], lr=lr)

    def matrices():
        edge = F.softplus(logits)
        weights = torch.eye(len(attr), dtype=torch.float32)
        weights[row, col] = edge
        weights[col, row] = edge
        inverse_degree = weights.sum(1).rsqrt()
        normalized = inverse_degree[:, None] * weights * inverse_degree[None, :]
        return edge, weights, normalized

    history = []
    for step in range(steps + 1):
        edge, weights, normalized = matrices()
        loss = F.mse_loss(normalized @ x, x) + 1e-3 * (edge - initial).pow(2).mean()
        history.append(float(loss.detach()))
        if step < steps:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return {
        'adjacency': normalized.detach().numpy().astype(np.float32),
        'raw_weights': weights.detach().numpy().astype(np.float32),
        'initial_raw_weights': raw.astype(np.float32),
        'objective_history': np.asarray(history),
        'edge_logits': logits.detach().numpy(),
    }


class DenseGCNAutoencoder(nn.Module):
    def __init__(self, dim, hidden=32):
        super().__init__()
        self.first = nn.Linear(dim, hidden)
        self.second = nn.Linear(hidden, dim)

    def forward(self, x, adjacency):
        hidden = F.relu(adjacency @ self.first(x))
        return torch.sigmoid(adjacency @ self.second(hidden))


class DenseGATAutoencoder(nn.Module):
    def __init__(self, dim, hidden=32):
        super().__init__()
        self.project = nn.Linear(dim, hidden, bias=False)
        self.attn_source = nn.Parameter(torch.empty(hidden))
        self.attn_target = nn.Parameter(torch.empty(hidden))
        self.decode = nn.Linear(hidden, dim)
        nn.init.xavier_uniform_(self.project.weight)
        nn.init.normal_(self.attn_source, std=0.1)
        nn.init.normal_(self.attn_target, std=0.1)

    def forward(self, x, adjacency):
        hidden = self.project(x)
        score = hidden @ self.attn_source[:, None] + (
            hidden @ self.attn_target[:, None]
        ).T
        score = F.leaky_relu(score, 0.2)
        mask = adjacency > 0
        attention = torch.softmax(score.masked_fill(~mask, -1e9), dim=1)
        aggregated = F.elu(attention @ hidden)
        return torch.sigmoid(self.decode(aggregated))


def train_refiner(model, attr, adjacency, seed, steps=1200, lr=0.01):
    set_seed(seed)
    x = torch.tensor(attr, dtype=torch.float32)
    graph = torch.tensor(adjacency, dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    for _ in range(steps):
        optimizer.zero_grad()
        output = model(x, graph)
        loss = F.mse_loss(output, x)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        return model(x, graph).cpu().numpy().astype(np.float32)


def build_prototypes(attr, sem, seed, beta, rho, topk):
    set_seed(seed)
    hybrid = build_class_graph(attr, sem, graph_mode="hybrid", topk=topk, rho=rho)
    attribute_graph = build_class_graph(
        attr, sem, graph_mode="attribute", topk=topk, rho=rho
    )
    learned = learn_adjacency(attr, seed)
    gcn_output = train_refiner(
        DenseGCNAutoencoder(attr.shape[1]), attr, hybrid, seed
    )
    gat_output = train_refiner(
        DenseGATAutoencoder(attr.shape[1]), attr, hybrid, seed
    )
    return {
        "RawAttr": attr,
        "AttributeGraph": np.clip(
            (1.0 - beta) * attr + beta * (attribute_graph @ attr), 0.0, 1.0
        ),
        "LearnedAdjacency": np.clip(
            (1.0 - beta) * attr + beta * (learned @ attr), 0.0, 1.0
        ),
        "GCNRefinement": np.clip(
            (1.0 - beta) * attr + beta * gcn_output, 0.0, 1.0
        ),
        "GATRefinement": np.clip(
            (1.0 - beta) * attr + beta * gat_output, 0.0, 1.0
        ),
        "HybridFixed": np.clip(
            (1.0 - beta) * attr + beta * (hybrid @ attr), 0.0, 1.0
        ),
    }, {
        "attribute": attribute_graph,
        "learned": learned,
        "hybrid": hybrid,
    }
