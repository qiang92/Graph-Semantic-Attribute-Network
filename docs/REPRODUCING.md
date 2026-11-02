# Reproducing the published-paper experiments

## Protocol

TEP tasks A–D use unseen sets F1/F6/F14, F4/F7/F10, F8/F11/F12 and F2/F3/F5. All other F1–F15 faults are seen. The signal branch receives 52 variables; the feature used by the attribute heads concatenates its latent embedding with the smoothed signal. The paper configuration uses hidden/latent dimensions 52/104, RMSprop learning rate 0.055, batch size 400, dropout 0.25, 900 epochs and evaluation every 10 epochs. The learning rate halves at epoch 450. Loss weights are 1, 1e-5, 1, 0.5 and 0.001 for attribute reconstruction, signal reconstruction, compatibility, off-diagonal correlation and center loss.

The hybrid graph uses rho=0.5, top-k=5 (other classes), self-loops, symmetric normalization and propagation beta=0.02. Attribute predictions are binary; graph-refined class prototypes are fractional. Reliability fits temporary GaussianNB heads on a stratified 80% seen-feature subset and measures errors on the remaining 20%. Inverse-MSE weights use epsilon=0.001, mean normalization and clipping at 2. Final diagnosis heads use all seen features.

TEP training comparisons select each variant's checkpoint by its highest unseen-test sample accuracy, retaining the first maximum. TPTS selects the highest unseen-test mean per-class accuracy. This is the SCE-based experimental selection protocol used for the reported comparisons. Fixed-prediction analyses retain the same trained predictions and change the matching rule. GZSFD uses the final training epoch; its score bias is calibrated using three pseudo-unseen folds drawn from seen classes.

TEP seeds are 7, 103, 904, 2026 and 3407. TPTS seeds are 42, 904, 2026, 7 and 103. Average tasks/splits within each seed first, then compute the mean and sample SD across the five seed averages. Do not pool all task scores to calculate the reported SD.

TPTS uses six seen-standardized current/voltage inputs and four G/C/B/A attributes. Training uses 250 epochs, batch size 256, hidden/latent dimensions 128/64, AdamW learning rate 0.001 and weight decay 0.0001. The primary result uses graph-guided representation training and uniform attribute matching; reliability weighting is a separate comparison.

## Experiment identifiers

Run `python run_experiment.py IDENTIFIER --output outputs/NAME`. All scientific values are explicit in `configs/experiments.json`.

| Paper experiment | Identifiers | Reference accuracy (%) |
|---|---|---|
| Primary TEP | `tep_main` | 82.39 ± 2.30 |
| Four component variants | `tep_base`, `tep_graph`, `tep_reliability`, `tep_main` | 78.59, 79.05, 80.62, 82.39 |
| Graph sources | `graph_attribute`, `graph_language`, `tep_main` | 80.56, 78.90, 82.39 |
| Prototype modules | `prototype_learned_adjacency`, `prototype_gcn`, `prototype_gat` | 81.37, 78.64, 77.60 |
| Language sensitivity | `language_deepseek`, `language_concise`, `language_nomic` | 80.66, 81.80, 81.44 |
| Propagation/weighting | `beta_0`, `beta_0.005`, `beta_0.01`, `beta_0.02`, `beta_0.03` | Both rules reported separately |
| Loss sensitivity | all `loss_…` identifiers | Full-precision reference CSV |
| Semantic integration | `semantic_direct`, `semantic_alignment` | 44.29, 78.44 |
| Graph removal | `edges_full46`, `edges_expert20`, `edges_random20` | 82.39, 81.49, 80.40 |
| Primary TPTS | `tpts_main` | 73.06 ± 1.25 |
| TPTS reliability | `tpts_weighted` | 72.69 ± 1.18 |
| Generalized ZSFD | `gzsfd` | H: 1.96 uncalibrated; 28.16 calibrated |
| Adapted SCE | `adapted_sce` | 79.75 ± 4.04 |

The random graph-removal arm pairs one of the five frozen masks with each seed. It retains twenty unordered edges plus the original self-loops, without renormalizing surviving weights. Its variability includes both graph and training randomness.

The semantic-integration families retain their recorded training configurations: direct prototype replacement uses 120 epochs (AdamW, hidden/latent 256/128), while latent alignment and primary GSAN use 900 epochs. Their objectives and architectures also differ; these are complete strategy comparisons, rather than single-module replacements. The 120-epoch value is retained to reproduce the reported 44.29% result.

## Fixed-prediction analyses

`analyze.py reliability` reports uniform matching, inverse-MSE, relation-conditioned MSE and Ledoit–Wolf Mahalanobis matching. Its reference means are 76.45, 82.39, 67.65 and 33.89%. It also recomputes per-task seen/unseen error correlations. Uniform 76.45% uses the predictions selected for weighted GSAN. The separately trained/selected graph-only ablation is 79.05%; these are different experiments.

`analyze.py entropy` computes entropy weights from seen calibration probabilities while keeping final Adapted SCE predictions binary and keeping its prototypes/checkpoint fixed. Its reference comparison is 79.75% versus 73.27%. It does not switch the diagnostic output to probability-valued attributes.

`analyze.py tpts_raw` uses signal-only GaussianNB attributes and seen-fitted standardization. It averages ten fault-pair macro accuracies and obtains 65.8362%. This deterministic reference is distinct from a learned signal-feature pipeline with graph/reliability disabled.

## Result identity

`reference_results/` contains results for experiments reported in the published paper and supplementary material, including their per-run and per-task records. `outputs/` contains new runs. The five-run pretrained TEP evaluator checks each task against its paper reference. TPTS pretrained models cover the ten splits of seed 42; their ten-split mean is 73.5387%. The paper's 73.0564% averages five runs. Reproduce that statistic with `tpts_main`, not by relabelling a single pretrained run.

Different hardware or library versions can change newly trained numerical trajectories. The tested versions are pinned in `requirements.txt`; fixed language inputs and pretrained inference provide separate reproducibility checks. No result from a new run is substituted for the archived main configuration automatically.

Published comparison methods in the main table, including the published SCE value of 76.5%, come from the cited studies. `adapted_sce` is the controlled local implementation used for subsequent experiments and is not claimed to reproduce that literature number.
