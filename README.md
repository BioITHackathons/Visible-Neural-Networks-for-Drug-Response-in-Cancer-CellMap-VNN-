# CellMap-VNN

Interpretable neural network for cancer clinical outcome prediction. The model structure mirrors a biological ontology hierarchy (NeST / Cell Map) so that predictions are explainable at the level of biological systems, not just individual genes.

This repo covers the full pipeline: data acquisition from cBioPortal, ontology-structured training, per-patient explainability, LLM-powered clinical interpretation, and an interactive web visualization.

**Related publications (please cite both if you use this repo):**
1. Park, S., Silva, E., Singhal, A. et al. *A deep learning model of tumor cell architecture elucidates response and resistance to CDK4/6 inhibitors.* Nat Cancer (2024). https://doi.org/10.1038/s43018-024-00740-1
2. Zhao, Singhal, et al. *Cancer Mutations Converge on a Collection of Protein Assemblies to Predict Resistance to Replication Stress.* Cancer Discov 14 (3): 508-523 (2024). https://doi.org/10.1158/2159-8290.CD-23-0641

## Pipeline

```
cBioPortal API
  |
  v
cbioport_download.py --> cbioport_transform.py
  |                         |
  v                         v
cbioportal_output/       nest_vnn_input/
                           (ontology, gene matrices, training_data.txt)
                            |
                            v
                         train.py --> model_final.pt
                            |
                            v
                         predict.py --> hidden embeddings (per-term, per-gene)
                            |
                            v
                         annotate_hierarchy.py
                            |
               +------------+-------------+
               |                          |
               v                          v
      RLIPP scores, gene scores,    patient_viz.html
      boolean logic, HTML viz             |
               |                          |
               v                          v
      interpretation pipeline       NeST-STRING Explorer
      (LLM clinical reasoning)     (Cytoscape.js hierarchy
               |                    + STRING PPI networks)
               v                          |
      SQLite DB (structured          <----+
      per-patient interpretations)        |
               |                          v
               +-------> Unified Flask app (port 5001)
                         Tab 1: Dashboard (population overview)
                         Tab 2: LLM Interpretation (per-patient)
                         Tab 3: NeST-STRING Explorer (interactive map)
```

## Quick start

### Environment

```bash
conda env create -f bioitworld_nest_vnn/conda-envs/environment.yml
conda activate nest_vnn
```

Requires Python 3.12, PyTorch 2.10. CUDA GPU recommended but CPU works (`-cuda cpu`).

### Training pipeline (interactive launcher)

```bash
cd bioitworld_nest_vnn
python scripts/run.py
```

The launcher walks through download, transform, train, predict, and annotate steps with menus and sensible defaults. To try the pipeline without a cBioPortal account, select "Download sample data" for the original GDSC drug-response dataset.

### LLM interpretation

```bash
cd bioitworld_nest_vnn
python interpretation/run_pipeline.py <study_id> <label> --limit 10 --batch_size 5
```

Requires either AWS credentials (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`) for Bedrock Claude or `OPENAI_API_KEY` for the OpenAI fallback.

### Web visualization

```bash
cd visualization/BioiThackCellMAP
pip install flask flask-cors requests
python server.py
```

Open http://localhost:5001. Three tabs:

| Tab | What it shows |
|-----|---------------|
| **Dashboard** | Population-level overview: patient counts, outcome distribution, NEST frequency, top recurrently important genes |
| **LLM Interpretation** | Per-patient clinical reasoning: NeST accordion cards with importance/RLIPP scores, biology and clinical text, gene alteration tables with targeted therapies |
| **NeST-STRING Explorer** | Interactive Cytoscape.js hierarchy map with STRING PPI network viewer and per-patient genomic context |

## Repository structure

```
cellmapvnn/
  bioitworld_nest_vnn/          <-- Training, prediction, and interpretation
    src/                        <-- Core model, training loop, RLIPP, annotation
    scripts/                    <-- CLI entry points (run.py, download, transform)
    interpretation/             <-- LLM pipeline: extract, batch, prompt, parse, store
    data/output/<study>/        <-- All per-study inputs and outputs
    conda-envs/                 <-- Environment specs
    mlflow.db                   <-- MLflow experiment tracking database
  visualization/
    BioiThackCellMAP/           <-- Flask server + frontend
      server.py                 <-- API proxy (NDEx, STRING, MyGene.info, patient data, LLM interpretations)
      patient_loader.py         <-- Parses patient_viz.html at startup
      static/                   <-- HTML/JS frontend (index, dashboard, interpretation, explorer)
```

See [`bioitworld_nest_vnn/README.md`](bioitworld_nest_vnn/README.md) for detailed documentation on training parameters, output files, explainability metrics (RLIPP, gene scores, boolean logic gates), and the interactive HTML visualizations.

See [`visualization/BioiThackCellMAP/README.md`](visualization/BioiThackCellMAP/README.md) for the NeST-STRING Explorer features and data requirements.

## How it works

Each patient is characterized by binary feature vectors for somatic mutations, copy number deletions, copy number amplifications, and optionally gene fusions. The neural network (`DrugCellNN`) is structured to mirror a biological ontology hierarchy:

- **Gene feature layers** compress each gene's multi-omic binary features to a scalar activation
- **Ontology term layers** are sparse linear layers where weight gradients are masked to enforce gene-term annotations from the hierarchy
- **Auxiliary output heads** at every ontology term are supervised against the same label as the root, ensuring gradients flow throughout the full hierarchy
- **RLIPP scores** measure how much predictive signal each system adds beyond its child systems, identifying which biological processes are most important for the model's predictions

The result is a model where activations at each node are interpretable as the state of a specific biological system, and importance can be attributed at both the system and gene level for individual patients.

## License

MIT License. See [`bioitworld_nest_vnn/LICENSE`](bioitworld_nest_vnn/LICENSE).
