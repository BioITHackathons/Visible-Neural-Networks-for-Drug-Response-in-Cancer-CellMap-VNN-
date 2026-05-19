# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

NeST-VNN (Nested Systems-level Visible Neural Network) predicts cancer patient clinical outcomes from tumor genotype. The neural network's structure mirrors a biological ontology hierarchy — each ontology term becomes a group of PyTorch modules — making predictions explainable at the level of biological systems. The main codebase lives in `bioitworld_nest_vnn/`.

## Environment setup

```bash
conda env create -f bioitworld_nest_vnn/conda-envs/environment.yml
conda activate nest_vnn
```

Requires Python 3.12, PyTorch 2.10. CUDA GPU recommended but CPU works (`-cuda cpu`).

## Common commands

All commands run from `bioitworld_nest_vnn/`.

```bash
# Interactive launcher (recommended — handles all steps)
python scripts/run.py

# Individual pipeline steps
python scripts/cbioport_download.py <study_id>
python scripts/cbioport_transform.py <study_id>

# Training
python src/train.py -onto data/output/<study_id>/nest_vnn_input/ontology.txt \
  -gene2id data/output/<study_id>/nest_vnn_input/gene2ind.txt \
  -cell2id data/output/<study_id>/nest_vnn_input/cell2ind.txt \
  -genotype data/output/<study_id>/nest_vnn_input/cell2mutation.txt \
  -cn_deletions data/output/<study_id>/nest_vnn_input/cell2cndeletion.txt \
  -cn_amplifications data/output/<study_id>/nest_vnn_input/cell2cnamplification.txt \
  -training_data data/output/<study_id>/nest_vnn_input/training_data.txt \
  -label <label_name> -task binary -modeldir data/output/<study_id>/<label>/model \
  -cuda 0

# LLM interpretation pipeline
python interpretation/run_pipeline.py <study_id> <label> --limit 10 --batch_size 5

# Interpretation dashboard
streamlit run interpretation/app.py
```

Commands run from `visualization/BioiThackCellMAP/`:

```bash
# NeST-STRING Explorer (interactive hierarchy + STRING PPI viewer)
pip install flask flask-cors requests
python server.py
# Open http://localhost:5001
```

## Architecture

### Data flow

```
cBioPortal API → cbioport_download.py → cbioportal_output/
  → cbioport_transform.py → nest_vnn_input/ (matrices + ontology + endpoints.json)
    → train.py → model_final.pt
      → predict.py → predictions + hidden embeddings (per-term, per-gene)
        → annotate_hierarchy.py → RLIPP scores, gene scores, boolean logic, HTML viz
          → interpretation/extractor.py → LLM pipeline → SQLite → Streamlit app
          → patient_viz.html → visualization/BioiThackCellMAP (NeST-STRING Explorer)
```

### Core modules (`src/`)

- **drugcell_nn.py** — `DrugCellNN(nn.Module)`: the ontology-structured network. Per-gene feature layers (Linear→tanh→BatchNorm compress multi-omic binary features to scalar) feed into sparse term layers where gradient masks enforce gene-term annotations from the ontology. Every term has auxiliary output heads for deep supervision.
- **training_data_wrapper.py** — Loads ontology DAG, gene/cell mappings, and stacks mutation/CNV/fusion matrices into the feature tensor. Handles train/val split and optional z-score normalization.
- **vnn_trainer.py** — Training loop with early stopping, auxiliary loss weighting (`-alpha`), and MLflow logging. Generates architecture schematic PNG.
- **train.py** — CLI entry point. Routes to `VNNTrainer` (direct) or `OptunaNNTrainer` (hyperparameter search with `-optimize 2`).
- **predict.py** — Inference: extracts per-term and per-gene hidden embeddings to `metrics/hidden/`.
- **rlipp_calculator.py / annotate_hierarchy.py** — Computes RLIPP scores (Ridge+PCA regression comparing parent vs. children predictive power), gene-level correlations, subsystem gene weights, boolean logic gate analysis, and per-patient importance. Produces interactive HTML visualizations.
- **util.py** — Z-score standardization, metric calculations (Pearson, Spearman, AUC, accuracy), input vector construction.

### Interpretation module (`interpretation/`)

Extracts per-patient results from MLflow artifacts, batches them, sends to an LLM (AWS Bedrock Claude or OpenAI fallback) for clinical interpretation, and stores structured results in SQLite (`nest_vnn_interpretations.db`). The Streamlit app (`app.py`) provides patient-level and population-level browsing.

- **extractor.py** — Harvests patient records from MLflow annotation artifacts
- **pipeline.py** — Orchestrates extract → batch → LLM → parse → store
- **llm_client.py** — Bedrock primary (Claude 3.5 Sonnet), OpenAI fallback; needs AWS_* or OPENAI_API_KEY env vars
- **prompts.py** — System prompt and batch message builder for clinical interpretation
- **db.py** — SQLite schema: `patients`, `patient_nests`, `patient_genes` tables

### NeST-STRING Explorer (`visualization/BioiThackCellMAP/`)

Interactive web tool for exploring the NeST hierarchy and STRING protein-protein interaction networks with per-patient genomic context. A Flask server proxies NDEx, STRING, and MyGene.info APIs to avoid CORS issues; the frontend is a single-page Cytoscape.js app.

- **server.py** — Flask proxy (port 5001). Endpoints: `/api/nest/<uuid>` (NDEx), `/api/string/network` (STRING PPI), `/api/mygene/<symbol>` (gene descriptions), `/api/patients/*` (patient data)
- **patient_loader.py** — Parses `patient_viz.html` at startup to build in-memory patient data (term importance, gene scores, predictions). Auto-discovers the file under `bioitworld_nest_vnn/data/output/` or `mlruns/`. Also loads binary alteration matrices (mutation/CNV/fusion) from `nest_vnn_input/` for gene badges.
- **static/index.html** — Single-page frontend: Cytoscape.js NeST hierarchy map on the left, STRING network panel on the right. Patient mode highlights top 3 directionally-significant NESTs (green = higher outcome, red = lower). Click any node to fetch its STRING PPI network.
- **nest_cli.py** — CLI utilities for NeST data

### Data layout

All outputs go under `data/output/<study_id>/`:
- `cbioportal_output/` — Raw downloads (mutations.csv, clinical_outcomes.csv, etc.)
- `nest_vnn_input/` — Transformed matrices, ontology, gene2ind, training_data.txt, endpoints.json, metadata.json
- `<label>/model/` — model_final.pt, std.txt (normalization params)
- `<label>/metrics/` — Predictions, probabilities, `hidden/` directory with per-term and per-gene embeddings
- `<label>/annotation/` — RLIPP scores, gene scores, boolean logic, GraphML/CX2 hierarchy, HTML visualizations, per-patient importance files

### MLflow artifact URIs

The MLflow `mlflow.db` stores absolute `file://` artifact URIs. These were originally recorded on Windows (`file:C:/Users/prath/...`) and must be updated when working on a different machine. The `_uri_to_path` function in `interpretation/extractor.py` handles both Windows and Unix URI formats.

### Key design decisions

- **Ontology-guided sparsity**: Weight gradients are masked to zero outside each term's annotated gene set — the hierarchy is a hard architectural constraint, not a soft regularizer.
- **Auxiliary supervision**: Every ontology term has output heads supervised against the same label as root (weighted by `-alpha`). This ensures gradients flow throughout the full hierarchy, not just root-to-leaf.
- **MSE over CCC for auxiliary loss**: Replaced Concordance Correlation Coefficient with MSE because CCC is numerically unstable when auxiliary heads produce near-constant predictions early in training.
- **RLIPP measures model organization, not ground truth**: RLIPP regresses embeddings against the model's own predictions, not against labels. Poor model performance means RLIPP scores describe an unreliable model.

### Key training flags

| Flag | Default | Purpose |
|---|---|---|
| `-task` | `continuous` | `binary` (BCEWithLogits) or `continuous` (MSE) |
| `-alpha` | `0.3` | Auxiliary supervision weight |
| `-optimize` | `1` | `1` = direct train, `2` = Optuna search then train |
| `-genotype_hiddens` | `4` | Hidden units per ontology term |
| `-zscore_method` | `auc` | `auc` = no normalization, `zscore`/`robustz` for continuous labels |
| `-patience` | `30` | Early stopping epochs |
| `-no_mlflow` | off | Disable MLflow tracking |
