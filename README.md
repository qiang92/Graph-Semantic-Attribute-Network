# GSAN inference

Inference code and 20 pretrained models for the four TEP tasks and five runs reported in the paper. Runs on CPU without an LLM service. Training and model-selection protocols are described in the paper and Supplementary Material.

## Install

Use Python 3.11 and run commands from this directory:

```sh
python -m pip install -r requirements.txt
```

## Reproduce the paper results

Prepare `d01.dat` through `d15.dat` in one directory. Use the paper's TEP records, each containing 480 rows and 52 signal columns in the original sensor order. Keep rows in their original temporal order. Data are supplied separately.

```sh
python evaluate.py --data /path/to/TEP_data --output results/tep
```

Expected result: **82.39% ± 2.30%**. The script evaluates all 20 models, averages the four tasks within each run, and reports the five-run mean and sample standard deviation.

Outputs in `results/tep`:

- `results.json`: overall and per-run results.
- `task_results.csv`: per-task accuracy and difference from the paper reference.
- `seed*_task_*.npz`: new predictions, attributes, distances, and true labels.

## Predict one signal record

```sh
python gsan_inference.py --model models/tep/seed904/task_1 --input /path/to/TEP_data/d01.dat --record-lengths 480 --output predictions.npz
```

Input can be a whitespace-delimited `.dat` file or a `.npy` array with 52 columns. Set `--record-lengths` to the length of each independent record. The code applies a trailing three-row mean separately within each record. Use `--preprocessed` only for inputs already processed in this way.

Each model predicts among its saved unseen fault classes. The output contains `labels`, `class_indices`, `attributes`, and `distances`.

## Files

- `gsan_inference.py`: preprocessing and prediction.
- `evaluate.py`: evaluation of all four tasks over five runs.
- `models/index.json`: model paths, task/run identifiers, and reference results.
- `models/tep/seed*/task_*/`: `encoder.pt` (encoder weights), `inference_arrays.npz` (attribute heads, prototypes, and weights), and `metadata.json` (model settings and class mapping).
- `assets/`: primary embeddings (15 × 1024), class graph (15 × 15), and propagated attributes (15 × 20), ordered F1–F15. These are provided for reference and are not required by the prediction command.
