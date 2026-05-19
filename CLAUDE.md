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

**Windows only — required before running:**
```powershell
$env:PYTHONUTF8=1
```
Without it, `cbioport_transform.py` crashes with a `UnicodeEncodeError` when writing `README.md` (the `≥` character is not encodable in the default cp1252 codec). Also, pressing Ctrl+C during the transform will print a harmless `FileNotFoundError: stty sane` — this is `run.py` trying a Unix terminal-reset command; safe to ignore on Windows.

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
python interpretation/run_pipeline.py <study_id> <label> --limit 10 --batch-size 5
# --offset N skips first N patients (useful to resume after a quota failure)
python interpretation/run_pipeline.py <study_id> <label> --limit 200 --offset 155 --no-skip

# Interpretation dashboard
streamlit run interpretation/app.py
```

Commands run from `visualization/BioiThackCellMAP/`:

```bash
# Unified visualization (3-tab Flask app: Dashboard, LLM Interpretation, NeST-STRING Explorer)
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
          → interpretation/extractor.py → LLM pipeline → SQLite
          → patient_viz.html → visualization/BioiThackCellMAP (unified 3-tab Flask app)
```

### Core modules (`src/`)

- **drugcell_nn.py** — `DrugCellNN(nn.Module)`: the ontology-structured network. Per-gene feature layers (Linear→tanh→BatchNorm compress multi-omic binary features to scalar) feed into sparse term layers where gradient masks enforce gene-term annotations from the ontology. Every term has auxiliary output heads for deep supervision.
- **training_data_wrapper.py** — Loads ontology DAG, gene/cell mappings, and stacks mutation/CNV/fusion matrices into the feature tensor. Handles train/val split and optional z-score normalization.
- **vnn_trainer.py** — Training loop with early stopping, auxiliary loss weighting (`-alpha`), and MLflow logging. Generates architecture schematic PNG.
- **train.py** — CLI entry point. Routes to `VNNTrainer` (direct) or `OptunaNNTrainer` (hyperparameter search with `-optimize 2`).
- **predict.py** — Inference: extracts per-term and per-gene hidden embeddings to `metrics/hidden/`.
- **rlipp_calculator.py / annotate_hierarchy.py** — Computes RLIPP scores (Ridge+PCA regression comparing parent vs. children predictive power), gene-level correlations, subsystem gene weights, boolean logic gate analysis, and per-patient importance. Produces interactive HTML visualizations.
- **util.py** — Z-score standardization, metric calculations (Pearson, Spearman, AUC, accuracy), input vector construction.

### Clinical feature integration

Clinical covariates (e.g. sample type, mutation count, fraction genome altered) can be optionally fed into the model alongside genomic features. The integration has three stages:

**1. Transform step (`cbioport_transform.py` / `run.py` → "Transform to NeST-VNN")**

After endpoint selection, the script prompts: *"Enter numbers to include as covariates"*. Any cBioPortal clinical attribute can be added. Encoding rules:
- **Numeric** (e.g. `MUTATION_COUNT`, `FRACTION_GENOME_ALTERED`): z-score normalized — mean and std saved to `covariate_stats.json` so values can be denormalized later.
- **Categorical/binary** (e.g. `SAMPLE_TYPE`): user-defined or auto-suggested integer mapping (e.g. `Metastasis=1`, `Primary=0`).

Each selected covariate becomes a `cov_<column_name>` column appended to `training_data.txt`. The `covariate_stats.json` file stores encoding parameters needed for display (the LLM extractor uses it to show human-readable values instead of z-scores).

**2. Model architecture (`drugcell_nn.py` / `training_data_wrapper.py`)**

- `training_data_wrapper.py` auto-detects all `cov_*` columns in `training_data.txt` and assembles a `(n_samples, n_covariates)` array alongside the genomic tensor.
- `drugcell_nn.py` adds a **clinical encoder** — a single `Linear → BatchNorm1d` layer that projects the covariates into the same dimensionality as the root ontology term's hidden state (`num_hiddens_genotype`).
- The root hidden state and the clinical embedding are **concatenated** before the final prediction layer (input size doubles from `h` to `2h` when covariates are present). This means clinical context gates the final prediction without touching the ontology-constrained layers.

**3. Architecture diagram (deterministic)**

Training automatically renders two PNGs in the model directory:
- `nn_architecture.png` — the **actual** ontology DAG wired into the network: every term node, every gene-term edge, the clinical encoder branch, and the final concatenation. This is generated deterministically from the ontology file, so it always matches the real model topology.
- `nn_schematic.png` — a simplified conceptual schematic (input → gene layers → term layers → root → prediction).

Both are logged to MLflow as artifacts.

### Interpretation module (`interpretation/`)

Extracts per-patient results from annotation artifacts, batches them, sends to an LLM for clinical interpretation, and stores structured results in SQLite (`nest_vnn_interpretations.db`).

**LLM used:** AWS Bedrock (Claude 3.5 Sonnet) is the primary backend; **OpenAI GPT-4o** is the fallback and is currently the active model (Bedrock credentials expired). The GPT-4o API generates the natural-language clinical text: pathway names, biological explanations, clinical reasoning, gene roles, and drug annotations. All LLM outputs were **manually validated** for biological plausibility before use.

**Top-3 NEST / gene selection rule:** Only NeSTs and genes with a statistically meaningful outcome direction are included in the LLM prompt. A NEST qualifies if `|mean_z| >= 0.2` AND `|pop_p_rho| >= 0.1`; a gene qualifies if `|signed_z| >= 0.5` AND `|pop_rho| >= 0.1`. This mirrors the NeST-STRING Explorer's `termDir` / `geneDir` logic.

- **extractor.py** — Harvests per-patient records: top-3 directional NESTs, top-3 directional genes per NEST, alteration types, covariates
- **pipeline.py** — Orchestrates extract → batch → LLM → parse → store; deletes stale nest/gene rows before each upsert
- **llm_client.py** — Bedrock primary, GPT-4o fallback; env vars needed: `OPENAI_API_KEY` (active) or `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY` + `AWS_DEFAULT_REGION`
- **prompts.py** — System prompt and batch message builder for clinical interpretation
- **db.py** — SQLite schema: `patients`, `patient_nests`, `patient_genes` tables

**Running the interpretation pipeline:**

```bash
# From bioitworld_nest_vnn/

# Full run (first time or after model update)
python interpretation/run_pipeline.py breast_msk_2025 binary_os_months --limit 200

# Re-process all patients with updated logic (overwrites existing records)
python interpretation/run_pipeline.py breast_msk_2025 binary_os_months --limit 200 --no-skip

# Resume after a quota failure at batch 32 (index 155)
python interpretation/run_pipeline.py breast_msk_2025 binary_os_months --limit 200 --offset 155 --no-skip
```

`--no-skip` forces reprocessing of already-stored patients. `--offset N` skips the first N patients so you can resume mid-run without re-sending patients that already succeeded.

### Unified visualization (`visualization/BioiThackCellMAP/`)

A single Flask app (port 5001) with an iframe-based 3-tab UI. Each tab is a self-contained HTML page loaded in an iframe — this isolates Cytoscape.js, Chart.js, and marked.js globals.

**Running the visualization locally:**

```bash
# From visualization/BioiThackCellMAP/
pip install flask flask-cors requests   # one-time
python server.py
# Open http://localhost:5001 in your browser
```

The server auto-discovers `patient_viz.html` from `bioitworld_nest_vnn/data/output/` (most recently modified) and reads LLM interpretations from `interpretation/nest_vnn_interpretations.db`. Both data sources must exist for all three tabs to show data. The Dashboard and LLM Interpretation tabs degrade gracefully (empty state) if the SQLite DB is missing.

**Tabs** (managed by `static/index.html` shell):
1. **Dashboard** (`static/dashboard.html`) — Population-level overview: patient counts, outcome distribution by risk class, top recurrently important NeSTs across the cohort, top recurrently important genes. Uses Chart.js v4. Data from the SQLite DB (`/api/interp/population/*`).
2. **LLM Interpretation** (`static/interpretation.html`) — Per-patient clinical interpretation: filterable patient table, NeST accordion cards showing importance/RLIPP scores, GPT-4o-generated biological explanation and clinical reasoning, gene tables with alteration badges and CFDE drug annotations. Full raw LLM markdown rendered via marked.js. Data from SQLite DB (`/api/interp/*`).
3. **NeST-STRING Explorer** (`static/explorer.html`) — Cytoscape.js NeST hierarchy map showing the full ontology with RLIPP-coloured nodes + STRING PPI viewer for selected NeSTs. In patient mode, shows which genes are altered and their outcome direction. Data from `patient_viz.html` in-memory (`/api/patient/<id>/*`) + NDEx/STRING/MyGene.info live APIs.

**Backend:**
- **server.py** — Flask proxy (port 5001). Two data sources:
  - In-memory patient data from `patient_viz.html` → `/api/patients/*`, `/api/patient/<id>/*`
  - SQLite LLM interpretations from `interpretation/db.py` → `/api/interp/*` (filters, patients, nests, genes, population aggregations)
  - NDEx/STRING/MyGene.info proxies → `/api/nest/*`, `/api/string/*`, `/api/gene/*`
- **patient_loader.py** — Parses `patient_viz.html` at startup to build in-memory patient data (term importance, mean z-scores, gene scores, predictions). Auto-discovers the file under `bioitworld_nest_vnn/data/output/` or `mlruns/`. Also loads binary alteration matrices (mutation/CNV/fusion) from `nest_vnn_input/` for gene badges.
- **nest_cli.py** — CLI utilities for NeST data

### Data layout

All outputs go under `data/output/<study_id>/`:
- `cbioportal_output/` — Raw downloads: `mutations.csv`, `fusions.csv`, `cnv.csv`, `clinical_outcomes.csv`, `patient_clinical.csv`
- `nest_vnn_input/` — Transformed matrices: `cell2ind.txt`, `gene2ind.txt`, `cell2mutation.txt`, `cell2cndeletion.txt`, `cell2cnamplification.txt`, `cell2fusion.txt`, `ontology.txt`, `training_data.txt`, `std.txt`, `endpoints.json`, `metadata.json`, `covariate_stats.json`
- `<label>/model/` — `model_final.pt`, `std.txt` (normalization params)
- `<label>/metrics/` — Predictions, probabilities, `hidden/` directory with per-term and per-gene embeddings
- `<label>/annotation/` — Full set of annotation outputs:
  - `rlipp_scores.txt` — RLIPP + P_rho + C_rho per ontology term (cross-cohort)
  - `gene_scores.txt` — Spearman correlation of gene embedding vs. predictions (cross-cohort)
  - `subsystem_gene_weights.txt` — Per-term gene rankings by PC1 correlation (all 815 term-gene pairs)
  - `top_subsystem_genes.txt` — Top 20 terms × top 5 driving genes
  - `boolean_logic.txt` — Boolean logic characterization of subsystem trios (AND/OR/NOR/NAND/etc.)
  - `top_systems.txt` — Top 20 systems by RLIPP
  - `hierarchy_annotated.graphml` — Annotated hierarchy (open in Cytoscape)
  - `hierarchy_annotated.cx2` — CX2 format (NDEx/Cytoscape Web)
  - `hierarchy_viz.html` — Interactive standalone hierarchy viewer
  - `patient_viz.html` — Interactive per-patient explainability viewer (drives the visualization app)
  - `patient_term_importance.txt` — Per-patient term importance (||z|| of hidden embedding, unsigned)
  - `patient_term_mean_z.txt` — Per-patient term mean z-score (signed; used for outcome direction)
  - `patient_rlipp.txt` — Per-patient RLIPP scores
  - `patient_gene_importance.txt` — Per-patient gene importance scores
  - `patient_gene_signed_z.txt` — Per-patient gene signed z-scores (used for outcome direction)

### Reference run: breast_msk_2025 / binary_os_months

The repo was developed and validated against this study. Key numbers for sanity-checking a fresh run:

**Dataset (cBioPortal study `breast_msk_2025`)**
- Source: MSK-IMPACT targeted sequencing, 3879 breast cancer samples, 3116 patients
- Molecular profiles: mutations (23 068 records), fusions (1 198), discrete CNA (1 962 774)
- Unique samples after dedup: 3784; training samples with valid label: 3777

**Transform choices**
- NDEx hierarchy UUID: `9a8f5326-aa6e-11ea-aaef-0ac135e8bacf` (default NeST)
- Gene frequency threshold: 0.5% → 430 genes; 407 annotated in ontology
- Min genes per assembly: 5 (paper default) → 103 surviving terms / 395 total, 117 hierarchy edges, 815 gene annotations
- Endpoint: `OS_MONTHS` → binary at threshold 46 months → label `binary_os_months`
  - class 0 (event/deceased): 2071 samples; class 1 (no-event/living): 1706 samples

**Training (CPU run, 200 epoch budget)**
- Params: batchsize=512, lr=0.001, alpha=0.3, patience=30, dropout=0.3, genotype_hiddens=4, task=binary
- Converged: early stopping at epoch 150 (val loss plateaued ~epoch 120)
- Test accuracy: **0.6794**

**Annotation**
- 103 terms × 407 genes × 3777 samples processed
- Top RLIPP systems: NEST:49 (5.185), NEST:80 (1.500), NEST:142 (1.452), NEST:14 (1.181, p_rho=0.363)
- Boolean logic: 51 trios matched; dominant gate: B_NOT_A (16), A_NOT_B (10), NOR (8)

**LLM interpretation**
- Pipeline run: `--limit 200 --no-skip` → 155/200 patients processed successfully (batches 32–40 hit OpenAI quota limit)
- Resumed with: `--limit 200 --offset 155 --no-skip`

### MLflow artifact URIs

The MLflow `mlflow.db` stores absolute `file://` artifact URIs. These were originally recorded on Windows (`file:C:/Users/prath/...`) and must be updated when working on a different machine:

```sql
-- Example: repoint artifact URIs to a new machine
sqlite3 bioitworld_nest_vnn/mlflow.db
UPDATE runs SET artifact_uri = REPLACE(artifact_uri, 'file:C:/Users/prath/OneDrive/Desktop/cellmapvnn', 'file:///Users/<you>/cellmapvnn');
```

The `_uri_to_path` function in `interpretation/extractor.py` handles both Windows and Unix URI formats.

### Key design decisions

- **Ontology-guided sparsity**: Weight gradients are masked to zero outside each term's annotated gene set — the hierarchy is a hard architectural constraint, not a soft regularizer.
- **Auxiliary supervision**: Every ontology term has output heads supervised against the same label as root (weighted by `-alpha`). This ensures gradients flow throughout the full hierarchy, not just root-to-leaf.
- **MSE over CCC for auxiliary loss**: Replaced Concordance Correlation Coefficient with MSE because CCC is numerically unstable when auxiliary heads produce near-constant predictions early in training.
- **RLIPP measures model organization, not ground truth**: RLIPP regresses embeddings against the model's own predictions, not against labels. Poor model performance means RLIPP scores describe an unreliable model.

### AI-assisted development

**Claude (Anthropic)** assisted in developing this codebase — specifically the data pipeline (`cbioport_download.py`, `cbioport_transform.py`, `run.py`), the LLM interpretation module (`interpretation/`), and exploratory data analysis (EDA) of the breast_msk_2025 cohort. All code outputs were **manually validated**: model training curves were inspected for convergence, RLIPP scores were cross-checked against known oncology biology, and LLM-generated clinical text was reviewed for plausibility before being stored in the database.

**OpenAI GPT-4o** is used at inference time to generate the natural-language interpretation text shown in the LLM Interpretation tab — pathway names, biological explanations, clinical reasoning per NeST, gene role descriptions, and drug annotations. GPT-4o is the active model (AWS Bedrock / Claude 3.5 Sonnet is the configured primary but is currently inactive due to expired credentials).

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
