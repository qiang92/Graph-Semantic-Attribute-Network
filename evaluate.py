"""Evaluate GSAN from raw TEP signals and summarize the five paper runs."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def accuracy(truth, prediction):
    return float((truth == prediction).mean() * 100)


def summarize(rows):
    seeds = list(dict.fromkeys(row['seed'] for row in rows))
    runs = [{'seed': seed, 'accuracy_percent': float(np.mean([
        row['accuracy_percent'] for row in rows if row['seed'] == seed]))} for seed in seeds]
    scores = [run['accuracy_percent'] for run in runs]
    return {'runs': len(runs), 'run_results': runs, 'mean_percent': float(np.mean(scores)),
            'sample_std_percent': float(np.std(scores, ddof=1))}


def load_benchmark(path):
    return [np.loadtxt(path / f'd{i:02d}.dat') for i in range(1, 16)]


def evaluate(data_path, models, *, batch_size=257, output=None):
    from gsan_inference import Predictor
    index = json.loads((models / 'index.json').read_text(encoding='utf-8'))
    entries = index['tep']['models']
    missing = [entry['path'] for entry in entries
               if not all((models / entry['path'] / name).is_file()
                          for name in ['metadata.json', 'encoder.pt', 'inference_arrays.npz'])]
    if missing:
        raise FileNotFoundError('Five-run evaluation requires the missing models: ' + ', '.join(missing))
    data = load_benchmark(data_path)
    rows = []
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        predictor = Predictor(models / entry['path'])
        candidates = predictor.metadata['candidate_indices']
        raw = np.vstack([data[label] for label in candidates])
        lengths = [len(data[label]) for label in candidates]
        truth = np.concatenate([np.full(length, label) for label, length in zip(candidates, lengths)])
        result = predictor.predict(raw, record_lengths=lengths, batch_size=batch_size)
        score = accuracy(truth, result['class_indices'])
        row = {'seed': entry['seed'], 'task': entry['task'], 'samples': len(truth),
               'accuracy_percent': score, 'paper_reference_percent': entry['paper_accuracy_percent'],
               'difference_pp': score - entry['paper_accuracy_percent']}
        rows.append(row)
        if output is not None:
            np.savez_compressed(output / f"seed{entry['seed']}_{entry['task']}.npz",
                                truth=truth, **result)
        print(json.dumps(row), flush=True)
    summary = {'dataset': 'tep', 'task_results': rows, **summarize(rows),
               'paper_reference': index['tep']['paper_summary'],
               'all_task_scores_match_paper': all(abs(row['difference_pp']) < 1e-10 for row in rows)}
    if output is not None:
        (output / 'results.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        with (output / 'task_results.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, type=Path, help='TEP directory containing d01.dat through d15.dat')
    parser.add_argument('--models', type=Path, default=Path(__file__).resolve().parent / 'models')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--batch-size', type=int, default=257)
    args = parser.parse_args()
    summary = evaluate(args.data, args.models, batch_size=args.batch_size, output=args.output)
    print(json.dumps({key: value for key, value in summary.items() if key != 'task_results'}, indent=2))


if __name__ == '__main__':
    main()
