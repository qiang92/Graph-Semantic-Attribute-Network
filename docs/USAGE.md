# Command reference

Run all commands below from the repository root.

# Graph Semantic Attribute Network (GSAN)

Code for **Large Language Model-Guided Graph Semantic Attribute Propagation for Zero-Shot Industrial Fault Diagnosis**, published in *Advanced Engineering Informatics*, **77** (2027), 105423. [Paper](https://doi.org/10.1016/j.aei.2026.105423).

The repository provides GSAN training, calibration, inference and the paper's experiment configurations for TEP and TPTS. Language inputs are prepared offline; the diagnostic pipeline runs with the supplied descriptions and vectors, without an LLM service.

## Install and prepare data

Use Python 3.11:

```sh
python -m pip install -r requirements.txt
```

The reference environment uses PyTorch 2.5.1 with CUDA 12.1, NumPy 2.4.4 and scikit-learn 1.8.0. CPU inference is supported. Obtain the measurement files as described in [data/README.md](../data/README.md):

```text
data/
  TEP_data/d01.dat ... d15.dat
  classData.csv
  TEP_attribute_matrix.npy
  semantic_cache/
```

## Evaluate pretrained TEP models

```sh
python evaluate.py --data data/TEP_data --output outputs/tep_pretrained
```

The twenty models cover four tasks and five runs. Expected accuracy is **82.39% ± 2.30%**, averaging the four tasks within each run before computing the mean and sample standard deviation across runs. `results.json` and `task_results.csv` contain the full-precision results.

## Train GSAN

```sh
python run_experiment.py tep_main --output outputs/tep_main
python run_experiment.py tpts_main --output outputs/tpts_main
```

These commands run all paper seeds and tasks at the paper training budget. For one complete TEP run:

```sh
python run_experiment.py tep_main --seeds 904 --output outputs/tep_seed904
```

The TPTS command covers all ten pairs among the five fault classes, keeping the normal class seen. Its five-run result is **73.06% ± 1.25%**. Each split uses mean per-class accuracy, and the ten split scores are averaged within a run.

The TEP/TPTS main runners save metrics, predictions and, for their default NB pathways, selected encoder/head models under `seed*/models/` (TEP) or `seed*/pair_*/models/` (TPTS). The full selected network state is also saved as `training_model.pt`. New output directories avoid combining old and new runs.

To supply measurements elsewhere, use `--data-dir /path/to/TEP_data` or `--tpts-csv /path/to/classData.csv`. `--epochs`, `--tep-tasks` and `--tpts-pairs` are development overrides; their reduced-budget or subset results are not the paper averages.

## Experiments and analysis

The complete configuration list is in [configs/experiments.json](../configs/experiments.json). Examples:

```sh
python run_experiment.py tep_base --output outputs/tep_base
python run_experiment.py prototype_gat --output outputs/gat
python run_experiment.py language_deepseek --output outputs/deepseek
python run_experiment.py tpts_weighted --output outputs/tpts_weighted
python run_experiment.py edges_expert20 --output outputs/expert20
python run_experiment.py gzsfd --output outputs/gzsfd
```

Fixed-prediction reliability comparisons can use the supplied five-run prediction/calibration caches:

```sh
python analyze.py reliability --output outputs/reliability
python analyze.py tpts_raw --output outputs/tpts_raw
```

After training `adapted_sce`, its frozen binary-prediction entropy comparison is:

```sh
python run_experiment.py adapted_sce --output outputs/adapted_sce
python analyze.py entropy --run-dir outputs/adapted_sce --output outputs/entropy
```

See [docs/REPRODUCING.md](../docs/REPRODUCING.md) for the experiment mapping, protocols and result identities. [reference_results/paper_results.csv](../reference_results/paper_results.csv) contains archived reference statistics, separate from newly generated outputs. The main paper's other published-method numbers are literature comparisons, not results claimed from these scripts.

## Predict a signal record

To evaluate all ten TPTS pretrained models from the primary seed-42 run:

```sh
python evaluate_tpts.py --data data/classData.csv --output outputs/tpts_pretrained
```

Its expected ten-split mean is 73.5387%; this is one run, while the paper's 73.0564% averages five runs.

```sh
python gsan_inference.py --model models/tep/seed904/task_1 --input data/TEP_data/d01.dat --record-lengths 480 --output predictions.npz
python tpts_inference.py --model models/tpts/pair_1_2 --input data/classData.csv --output tpts_predictions.npz
```

Each model predicts only among its saved candidate faults. Passing a complete TPTS CSV predicts every row against that pair; this does not automatically calculate benchmark accuracy. The TPTS models represent one primary-configuration run (seed 42), not the five-run mean.

## Optional language generation

Frozen inputs reproduce the published language comparisons. To prepare a new input set with a local Ollama service:

```sh
python prepare_language.py --output outputs/new_language
python prepare_language.py --encoder nomic-embed-text --reuse-primary-texts --output outputs/new_nomic
```

Original prompts, texts and fixed vectors are supplied. Newly generated text can differ from the archived text; use the fixed inputs for the reported diagnostic comparisons.

## Sources

The implementation builds on [SCE](https://github.com/htz-ecust/Semantic-consistent-Embedding). The adapted baseline and GSAN changes are distinguished in the experiment descriptions. [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) contains the source attribution and the original SCE MIT license. The GSAN license is in [LICENSE](../LICENSE).
