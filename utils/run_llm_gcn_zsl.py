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

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
        self.attr_proto = nn.Sequential(
            nn.Linear(attr_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, latent),
        )
        self.gcn1 = GCNLayer(hidden, hidden)
        self.gcn2 = GCNLayer(hidden, latent)
        self.attr_head = nn.Linear(latent, attr_dim)
        self.sem_head = nn.Linear(latent, sem_dim)
        self.recon_x = nn.Linear(latent * 2, input_dim)
        self.recon_attr = nn.Linear(latent * 2, attr_dim)
        self.bn_z = nn.BatchNorm1d(latent, affine=False)

    def encode_signal(self, x):
        return F.normalize(self.signal(x), dim=1)

    def encode_prototypes(self, attr_proto, sem_proto, adj, proto_blend=0.35):
        h = self.proto_in(torch.cat([attr_proto, sem_proto], dim=1))
        h = F.relu(self.gcn1(h, adj))
        gcn_proto = self.gcn2(h, adj)
        attr_proto_z = self.attr_proto(attr_proto)
        fused = (1.0 - proto_blend) * attr_proto_z + proto_blend * gcn_proto
        return F.normalize(fused, dim=1)

    def forward(self, x, attr_proto, sem_proto, adj):
        z = self.encode_signal(x)
        p = self.encode_prototypes(attr_proto, sem_proto, adj)
        attr_pred = self.attr_head(z)
        sem_pred = self.sem_head(z)
        return z, p, attr_pred, sem_pred

    def reconstruct(self, z, p_y):
        return self.recon_x(torch.cat([z, p_y], dim=1)), self.recon_attr(torch.cat([p_y, z], dim=1))

    def compatibility_loss(self, z, p_y, offdiag_weight=0.5):
        n = z.shape[0]
        c = self.bn_z(z).T @ self.bn_z(p_y) / n
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = c.flatten()[:-1].view(c.shape[0] - 1, c.shape[0] + 1)[:, 1:].flatten().pow_(2).sum()
        return on_diag + offdiag_weight * off_diag
