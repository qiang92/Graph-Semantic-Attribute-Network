"""Standalone GSAN inference. Inputs and class identifiers are documented in README.md."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def smooth_records(x, lengths, window=3):
    x = np.asarray(x, dtype=np.float64)
    result = x.copy()
    start = 0
    if sum(lengths) != len(x):
        raise ValueError('Record lengths must sum to the number of input rows.')
    for length in lengths:
        for row in range(start + window - 1, start + length):
            result[row] = x[row - window + 1:row + 1].mean(axis=0)
        start += length
    return result


def predict_attributes(features, heads):
    columns = []
    for head in heads:
        if 'constant' in head:
            columns.append(np.full(len(features), head['constant']))
        else:
            joint = []
            for prior, mean, variance in zip(head['prior'], head['theta'], head['var']):
                score = np.log(prior) - 0.5 * np.sum(np.log(2.0 * np.pi * variance))
                score = score - 0.5 * np.sum((features - mean) ** 2 / variance, axis=1)
                joint.append(score)
            columns.append(head['classes'][np.argmax(np.asarray(joint).T, axis=1)])
    return np.stack(columns, axis=1)


class TEPEncoder(nn.Module):
    def __init__(self, input_dim, hidden, latent):
        super().__init__()
        self.x1 = nn.Linear(input_dim, hidden)
        self.x2 = nn.Linear(hidden, latent)
        self.bn_hidden_x = nn.BatchNorm1d(hidden)
        self.ln_hidden = nn.LayerNorm(hidden)

    def forward(self, x):
        return self.x2(F.relu(self.ln_hidden(self.bn_hidden_x(self.x1(x)))))


class TPTSEncoder(nn.Module):
    def __init__(self, input_dim, hidden, latent):
        super().__init__()
        self.signal = nn.Sequential(nn.Linear(input_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(0.0), nn.Linear(hidden, latent))

    def forward(self, x):
        return F.normalize(self.signal(x), dim=1)


class Predictor:
    def __init__(self, model_dir):
        model_dir = Path(model_dir)
        self.metadata = json.loads((model_dir / 'metadata.json').read_text(encoding='utf-8'))
        self.arrays = dict(np.load(model_dir / 'inference_arrays.npz', allow_pickle=False))
        kind = self.metadata['dataset']
        cls = TEPEncoder if kind == 'TEP' else TPTSEncoder
        self.encoder = cls(**self.metadata['encoder'])
        self.encoder.load_state_dict(torch.load(model_dir / 'encoder.pt', map_location='cpu', weights_only=True))
        self.encoder.eval()
        self.heads = []
        for j, info in enumerate(self.metadata['attribute_heads']):
            self.heads.append(info if 'constant' in info else {key: self.arrays[f'head_{j}_{key}'] for key in ['classes', 'theta', 'var', 'prior']})

    def preprocess(self, x, record_lengths=None):
        x = np.asarray(x)
        if x.ndim == 1:
            x = x[None, :]
        if x.ndim != 2 or x.shape[1] != self.metadata['encoder']['input_dim']:
            raise ValueError('Input must have shape (samples, input_dim); see metadata.json.')
        if self.metadata['dataset'] == 'TEP':
            if record_lengths is None:
                raise ValueError('Raw TEP input requires explicit record_lengths, including a new single-row record.')
            return smooth_records(x, record_lengths, self.metadata['preprocessing']['window']).astype(np.float32)
        # Match StandardScaler.transform on the original float32 TPTS rows:
        x = np.asarray(x, dtype=np.float32).copy()
        x -= self.arrays['scaler_mean'].astype(x.dtype)
        x /= self.arrays['scaler_scale'].astype(x.dtype)
        return x

    def predict(self, x, *, preprocessed=False, record_lengths=None, batch_size=0):
        x = np.asarray(x, dtype=np.float32) if preprocessed else self.preprocess(x, record_lengths)
        if x.ndim == 1:
            x = x[None, :]
        if x.ndim != 2 or x.shape[1] != self.metadata['encoder']['input_dim']:
            raise ValueError('Unexpected input shape.')
        if batch_size <= 0:
            batch_size = len(x)
        attr_parts = []
        with torch.inference_mode():
            for start in range(0, len(x), batch_size):
                bx = torch.from_numpy(x[start:start + batch_size])
                features = torch.cat([self.encoder(bx), bx], dim=1).numpy()
                attr_parts.append(predict_attributes(features, self.heads))
        attributes = np.concatenate(attr_parts)
        if self.metadata['dataset'] == 'TPTS':
            attributes = attributes.astype(np.float32)
        distances = (((attributes[:, None, :] - self.arrays['prototypes'][None, :, :]) ** 2) * self.arrays['weights'][None, None, :]).sum(axis=2)
        indices = self.arrays['candidate_indices'][distances.argmin(axis=1)]
        labels = np.array([self.metadata['class_names'][str(int(i))] for i in indices])
        return {'class_indices': indices, 'labels': labels, 'attributes': attributes, 'distances': distances}


def load_input(path):
    path = Path(path)
    if path.suffix == '.npy':
        return np.load(path, allow_pickle=False)
    if path.suffix == '.csv':
        columns = ['Ia', 'Ib', 'Ic', 'Va', 'Vb', 'Vc']
        with path.open(encoding='utf-8-sig', newline='') as stream:
            return np.asarray([[float(row[key]) for key in columns] for row in csv.DictReader(stream)], dtype=np.float32)
    return np.loadtxt(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--input', required=True, type=Path, help='.npy array, whitespace-delimited .dat, or named TPTS .csv')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--preprocessed', action='store_true')
    parser.add_argument('--record-lengths', type=int, nargs='+')
    parser.add_argument('--batch-size', type=int, default=0)
    args = parser.parse_args()
    data = load_input(args.input)
    result = Predictor(args.model).predict(data, preprocessed=args.preprocessed, record_lengths=args.record_lengths, batch_size=args.batch_size)
    np.savez_compressed(args.output, **result)
    print(json.dumps({'samples': len(result['labels']), 'predicted_labels': np.unique(result['labels']).tolist(), 'output': args.output.name}))


if __name__ == '__main__':
    main()
