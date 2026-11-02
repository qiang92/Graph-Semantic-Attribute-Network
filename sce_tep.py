import os

import numpy as np

import pandas as pd

import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

from torch.utils.data import DataLoader, Dataset

from torch.optim.lr_scheduler import MultiStepLR

from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC, LinearSVC

CONFIG = {
    "model_mode": "NB",            # rf / NB / SVC_linear / FS
    "batch_size": 400,
    "lr": 5.5e-2,                 # lr (BEST: e52)
    "epochs": 900,                 # epochs 900
    "dim": [52, 104],              # [Hidden, Latent]
    "lambda_": [1.0, 1e-5, 1.0, 0.5], # [R1, R2, CM, Off-diag]
    "dropout": 0.25,               # dropout 0.25
    "tau": 0.07,                   # Temperature for RankCLIP logic
    "rank_clip_coef": 0.0,         # RankCLIP disabled
    "center_coef": 0.001,          # Center Loss coefficient
    # Evolution 12.0: Dual-Stream
    "dual_stream": False,          # Enable dual-stream architecture
    "fusion_mode": "concat", # concat / weighted_add
    "sma_window": 3,               # SMA for Stream B
    "seed": 904,
    "sam_rho": 0.05,              # SAM neighborhood size
    "sam_enabled": False,          # Disable SAM
}

TEST_TASKS = [
    {'test_class': [1, 6, 14], 'seen_classes': [2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 15]},
    {'test_class': [4, 7, 10], 'seen_classes': [1, 2, 3, 5, 6, 8, 9, 11, 12, 13, 14, 15]},
    {'test_class': [8, 11, 12], 'seen_classes': [1, 2, 3, 4, 5, 6, 7, 9, 10, 13, 14, 15]},
    {'test_class': [2, 3, 5], 'seen_classes': [1, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]}
]

class MyDataset(Dataset):
    def __init__(self, data, attr, label):
        self.data = data
        self.attr = attr
        self.label = label
    def __len__(self): return self.data.shape[0]
    def __getitem__(self, idx): return self.data[idx], self.attr[idx], self.label[idx]

def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

class CenterLossModule(nn.Module):
    """Learnable class centers in latent space - crucial for Task C (Random Variation)"""
    def __init__(self, num_classes, feat_dim):
        super(CenterLossModule, self).__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
    def forward(self, features, labels):
        # features: (N, D), labels: (N,) of class indices
        batch_centers = self.centers[labels]  # (N, D)
        loss = ((features - batch_centers) ** 2).sum(dim=1).mean()
        return loss

class Embedding_Net(nn.Module):
    def __init__(self, dim, lambda_, dropout=0.1, dual_stream=False, fusion_mode='concat'):
        super(Embedding_Net, self).__init__()
        self.dual_stream = dual_stream
        self.fusion_mode = fusion_mode

        if dual_stream:
            # Path A (Sharp): 52-dim raw signal -> 64-dim
            self.l11_a = nn.Linear(52, 64)
            # Path B (Stable): 33-dim filtered signal -> 64-dim
            self.l11_b = nn.Linear(33, 64)
            if fusion_mode == 'concat':
                # Fusion: Concat(64+64=128) -> dim[1]
                self.l12 = nn.Linear(128, dim[1])
            elif fusion_mode == 'weighted_add':
                # Fusion: Weighted Add (64) -> dim[1]
                self.l12 = nn.Linear(64, dim[1])
                # Learnable weights for each path
                self.stream_weight = nn.Parameter(torch.ones(2) / 2)
        else:
            # Original single-stream
            self.l11 = nn.Linear(dim[0], dim[0])
            self.l12 = nn.Linear(dim[0], dim[1])
        self.l13 = nn.Linear(2 * dim[1], 52)  # Reconstruction (uses dim[0]=52 for raw input)
        # Attribute encoder (unchanged)
        self.l21 = nn.Linear(20, dim[0])
        self.l22 = nn.Linear(dim[0], dim[1])
        self.l23 = nn.Linear(2 * dim[1], 20)
        self.bn1 = nn.BatchNorm1d(dim[0])
        self.ln1 = nn.LayerNorm(dim[0])
        self.bn2 = nn.BatchNorm1d(dim[1])
        self.ln2 = nn.LayerNorm(dim[1])
        self.dropout = nn.Dropout(dropout)
        self.lambda_ = lambda_
    def transform(self, x, a):
        """Extract features - supports dual-stream"""
        if self.dual_stream:
            # x is (x_raw, x_filtered) tuple
            x_raw, x_filtered = x
            # Path A: raw signal
            h1_a = F.relu(self.l11_a(x_raw))
            # Path B: filtered signal
            h1_b = F.relu(self.l11_b(x_filtered))
            # Fusion: based on mode
            if self.fusion_mode == 'concat':
                h1 = torch.cat([h1_a, h1_b], dim=1)
            elif self.fusion_mode == 'weighted_add':
                # Normalize weights to sum to 1
                w = F.softmax(self.stream_weight, dim=0)
                h1 = w[0] * h1_a + w[1] * h1_b
            z1 = self.l12(self.dropout(h1))
        else:
            h1 = self.bn1(self.l11(x))
            z1 = self.l12(self.dropout(F.relu(self.ln1(h1))))
        # Attribute path (unchanged)
        h2 = self.bn1(self.l21(a))
        z2 = self.l22(self.dropout(F.relu(self.ln1(h2))))
        return z1, z2
    def reconstruction(self, z1, z2):
        """重构原始输入的方法"""
        f1 = torch.cat([z1, z2], dim=1)
        f2 = torch.cat([z2, z1], dim=1)
        x_ = self.l13(f1)
        a_ = torch.sigmoid(self.l23(f2))
        return x_, a_
    def compability_loss(self, z1, z2):
        N, D = z1.shape
        c = self.bn2(z1).T @ self.bn2(z2) / N
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        return on_diag + self.lambda_[3] * off_diag
    def focal_compatibility_loss(self, z1, z2, gamma=2.0):
        """Focal Compatibility Loss - focuses on hard samples"""
        N, D = z1.shape
        c = self.bn2(z1).T @ self.bn2(z2) / N  # (D, D)
        # Create target: on-diagonal should be 1, off-diagonal should be 0
        target = torch.eye(N, device=c.device)
        # Compute per-sample losses
        per_sample_loss = (c - target).pow(2).sum(dim=1)
        # Focal weighting: down-weight easy samples
        focal_weight = (1 - torch.exp(-per_sample_loss)).pow(gamma)
        loss = (focal_weight * per_sample_loss).mean()
        return loss
    def rank_consistency_loss(self, z1, z2, tau=0.07):
        # RankCLIP: symmetric KL-Divergence on similarity matrix
        S = torch.matmul(z1, z2.T) / tau  # similarity matrix
        # Clamp for numerical stability
        S = torch.clamp(S, min=-50, max=50)
        P = F.softmax(S, dim=1)  # row-wise softmax
        Q = F.softmax(S, dim=0)  # column-wise softmax
        # Add small epsilon for numerical stability
        eps = 1e-8
        P = P + eps
        Q = Q + eps
        P = P / P.sum(dim=1, keepdim=True)
        Q = Q / Q.sum(dim=0, keepdim=True)
        # Symmetric KL: 0.5 * (KL(P||Q) + KL(Q||P))
        kl1 = F.kl_div(P.log(), Q, reduction='batchmean')
        kl2 = F.kl_div(Q.log(), P, reduction='batchmean')
        return 0.5 * (kl1 + kl2)
    def centering_loss(self, z1, z2):
        # Center embeddings around zero for better cross-modal alignment
        center_z1 = z1 - z1.mean(dim=0, keepdim=True)
        center_z2 = z2 - z2.mean(dim=0, keepdim=True)
        return center_z1.pow(2).mean() + center_z2.pow(2).mean()
    def forward(self, x, a):
        z1, z2 = self.transform(x, a)
        x_, a_ = self.reconstruction(z1, z2)
        r1 = self.lambda_[0] * F.mse_loss(a, a_)
        r2 = self.lambda_[1] * F.mse_loss(x, x_)
        cm = self.lambda_[2] * self.compability_loss(z1, z2)
        rc = CONFIG["rank_clip_coef"] * self.rank_consistency_loss(z1, z2, CONFIG["tau"])
        ct = CONFIG["center_coef"] * self.centering_loss(z1, z2)
        loss = r1 + r2 + cm + rc + ct
        return {"z1": z1, "z2": z2, "loss": loss, "r1": r1, "cm": cm}

def pre_model_eval(model_name, traindata, train_attr_label, testdata, testlabel, attribute_matrix):
    model_dict = {
        'rf': RandomForestClassifier(n_estimators=100),
        'NB': GaussianNB(),
        'SVC_linear': SVC(kernel='linear'),
        'FS': LinearSVC()
    }
    clf = model_dict[model_name]
    res_list = []
    for i in range(train_attr_label.shape[1]):
        if np.max(train_attr_label[:, i]) != 0:
            clf.fit(traindata, train_attr_label[:, i])
            res = clf.predict(testdata)
        else:
            res = np.zeros(testdata.shape[0])
        res_list.append(res)
    
    test_pre_attr = np.stack(res_list, axis=1)
    unique_labels = np.unique(testlabel)
    
    label_pred = []
    for i in range(test_pre_attr.shape[0]):
        dist = np.sum((attribute_matrix - test_pre_attr[i, :])**2, axis=1)
        label_pred.append(unique_labels[dist.argmin()])
    
    return accuracy_score(label_pred, testlabel)
