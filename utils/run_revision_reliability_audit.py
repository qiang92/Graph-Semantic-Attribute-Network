import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.covariance import LedoitWolf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.run_llm_gcn_zsl import normalize_rows


def inverse_weights(mse, eps=1e-3, clip=2.0):
    weights = 1.0 / (np.asarray(mse, dtype=np.float64) + eps)
    weights /= weights.mean() + 1e-12
    return np.clip(weights, 0.0, clip)


def weighted_predict(pred_attr, prototypes, labels, weights):
    dist = (
        (pred_attr[:, None, :] - prototypes[None, :, :]) ** 2
        * weights[None, None, :]
    ).sum(axis=2)
    return labels[np.argmin(dist, axis=1)]


def mahalanobis_predict(pred_attr, prototypes, labels, precision):
    delta = pred_attr[:, None, :] - prototypes[None, :, :]
    dist = np.einsum("nck,kl,ncl->nc", delta, precision, delta)
    return labels[np.argmin(dist, axis=1)]


def class_conditional_predict(pred_attr, prototypes, labels, candidate_weights):
    distance = (
        (pred_attr[:, None, :] - prototypes[None, :, :]) ** 2
        * candidate_weights[None, :, :]
    ).sum(axis=2)
    return labels[np.argmin(distance, axis=1)]


def safe_correlation(x, y, method):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return np.nan, np.nan
    result = pearsonr(x, y) if method == "pearson" else spearmanr(x, y)
    return float(result.statistic), float(result.pvalue)
