# Repository contents

| Path | Purpose |
|---|---|
| `README.md`, `README.zh-CN.md` | English/Chinese project home pages with framework, key results and quick-start commands |
| `docs/images/` | Framework and result figures rendered from the paper's submitted PDF figure sources |
| `docs/USAGE.md` | Detailed command reference |
| `requirements.txt` | Versions used for the release checks |
| `run_experiment.py` | Run a named configuration, task/seed loops and statistical aggregation |
| `configs/experiments.json` | Thirty-six paper experiment configurations |
| `sce_tep.py` | SCE-derived embedding components and TEP training constants |
| `utils/run_gsan_tep.py` | TEP representation training, graph propagation, calibration and diagnosis |
| `utils/run_tpts_llm_gcn_zsl.py` | TPTS representation training and attribute diagnosis |
| `utils/export_model.py` | Save selected NB diagnosis models without changing scientific settings |
| `utils/run_llm_gcn_zsl.py` | Shared graph functions and TPTS signal/prototype network components |
| `utils/read_tep_data.py` | TEP loading, per-record smoothing and task data assembly |
| Other `utils/run_*.py` | Adapted SCE, prototype alternatives, semantic strategies, GZSFD and edge removal |
| `analyze.py` | Fixed-prediction reliability, binary entropy and signal-only TPTS comparisons |
| `evaluate.py`, `gsan_inference.py` | Existing five-run TEP pretrained evaluation and inference |
| `evaluate_tpts.py`, `tpts_inference.py` | Ten-pair TPTS pretrained evaluation and inference |
| `models/tep/` | Twenty pretrained models for the five paper TEP runs |
| `models/tpts/` | Ten models from the final-protocol TPTS seed-42 run |
| `assets/` | Primary TEP embeddings, graph and prototypes retained from the inference release |
| `data/` | Engineering attributes, fixed texts, vectors, prompts and graph-removal masks; measurement download instructions |
| `prepare_language.py` | Optional language generation/encoding through a local Ollama endpoint |
| `reference_results/` | Published-paper and supplementary results, per-run/task records and saved prediction/calibration arrays |
| `docs/REPRODUCING.md` | Scientific settings, experiment-to-paper mapping and result identities |
| `LICENSE`, `THIRD_PARTY_NOTICES.md` | GSAN MIT license, SCE source attribution and original SCE MIT license |
| `CITATION.cff`, `CITATION.bib` | Author/repository metadata and the final paper citation |

This repository excludes editorial correspondence, reviewer identities, personal configurations, notification credentials, development logs and unrelated follow-up research. New run outputs belong in `outputs/` and are ignored by Git.
