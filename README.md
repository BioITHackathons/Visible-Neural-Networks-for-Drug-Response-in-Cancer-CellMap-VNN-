# NeST-VNN: interpretable outcome prediction for cancer patients

NeST-VNN is a visible neural network that predicts clinical outcomes (survival, recurrence, drug response) from tumor genotype. The network is wired to follow the NeST hierarchy of protein assemblies, so each prediction can be traced back to the biological systems and genes that drove it.

Inputs are binary per-gene features: somatic mutations, copy-number deletions and amplifications, and optionally gene fusions. Clinical covariates such as sample type or mutation count can be added.

This repo was built at the Bio-IT World hackathon. It adds a cBioPortal data pipeline, per-patient explainability, LLM-written interpretations and a web dashboard on top of [idekerlab/nest_vnn](https://github.com/idekerlab/nest_vnn).

- **Live demo:** https://d1zrqe6xve7yby.cloudfront.net
- **Video walkthrough:** https://youtu.be/bjEPVhkSdC0
- **Slides:** https://view.genially.com/6a0ba1856731f1a27bb6d625

## Repository layout

| Path | Contents |
|---|---|
| `bioitworld_nest_vnn/scripts/` | cBioPortal download and transform, plus the interactive launcher `run.py` |
| `bioitworld_nest_vnn/src/` | Model (`drugcell_nn.py`), training, prediction and annotation |
| `bioitworld_nest_vnn/interpretation/` | LLM interpretation pipeline and its SQLite results |
| `bioitworld_nest_vnn/data/output/breast_msk_2025/` | A complete reference run: inputs, trained model, annotations |
| `visualization/BioiThackCellMAP/` | Flask web app: dashboard, LLM interpretations, NeST-STRING explorer |
| `docs/` | Output and model reference |

## Setup

```bash
cd bioitworld_nest_vnn
conda env create -f conda-envs/environment.yml
conda activate nest_vnn
```

This installs Python 3.12 and PyTorch 2.10. A CUDA GPU speeds up training, but everything also runs on CPU (`-cuda cpu`). On Windows, set `$env:PYTHONUTF8=1` before running the transform step.

## Running the pipeline

All commands run from `bioitworld_nest_vnn/`. The simplest way in is the interactive launcher, which walks through each step and prompts for parameters:

```bash
python scripts/run.py
```

The steps it runs, and the scripts behind them:

1. **Download** a cBioPortal study: `python scripts/cbioport_download.py breast_msk_2025`
2. **Transform** it into model inputs: `python scripts/cbioport_transform.py breast_msk_2025`. This step is interactive: you choose the outcome endpoint, the gene-frequency cutoff, the NDEx hierarchy, and any clinical covariates.
3. **Train** with `src/train.py`.
4. **Predict** with `src/predict.py`, which writes predictions and the hidden state of every term and gene.
5. **Annotate** with `src/annotate_hierarchy.py`, which computes RLIPP scores, gene scores, Boolean logic fits and per-patient importance, and writes the HTML viewers.

Outputs go to `data/output/<study_id>/`. To try the model without cBioPortal, choose **Download sample data** in `run.py`. It fetches the original GDSC drug-response dataset.

Training example (the launcher builds this command for you):

```bash
S=data/output/breast_msk_2025/nest_vnn_input
M=data/output/breast_msk_2025/binary_os_months/model
python src/train.py -onto $S/ontology.txt -gene2id $S/gene2ind.txt -cell2id $S/cell2ind.txt \
  -train $S/training_data.txt -mutations $S/cell2mutation.txt \
  -cn_deletions $S/cell2cndeletion.txt -cn_amplifications $S/cell2cnamplification.txt \
  -label binary_os_months -task binary -std $M/std.txt -modeldir $M -cuda cpu
```

MLflow tracking is on by default; pass `-no_mlflow` to turn it off.

## Outputs

Each run writes three folders under `data/output/<study_id>/<label>/`:

- `model/` holds the trained checkpoint (`model_final.pt`).
- `metrics/` holds predictions and hidden states.
- `annotation/` holds the explainability results. The main files are `rlipp_scores.txt` (how much each system adds beyond its children), `gene_scores.txt`, `boolean_logic.txt`, and two viewers that open in any browser: `hierarchy_viz.html` and `patient_viz.html`.

See [docs/outputs.md](docs/outputs.md) for every file and how to read the scores, and [docs/model.md](docs/model.md) for the network design, training flags and code layout.

RLIPP and the gene scores are measured against the model's own predictions, not the true labels. They show what the model relies on, so read them alongside its test performance.

## Reference run

`data/output/breast_msk_2025/binary_os_months/` holds a complete run on MSK-IMPACT breast cancer data from cBioPortal (`breast_msk_2025`):

| | |
|---|---|
| Samples | 3,777 |
| Genes | 407 |
| NeST systems | 103 |
| Label | overall survival above or below 46 months |
| Validation accuracy | 0.69, on a random 20% of samples held out during training |

There is no separate test set. Predictions in `metrics/` cover all samples, training ones included; their accuracy, 0.71, is not a test score.

## LLM interpretation

`interpretation/` turns each patient's top systems and genes into a plain-language summary and stores the results in `interpretation/nest_vnn_interpretations.db`. It calls AWS Bedrock first and falls back to OpenAI. Set `OPENAI_API_KEY`, or AWS credentials, before running it.

```bash
python interpretation/run_pipeline.py breast_msk_2025 binary_os_months --limit 200
```

Use `--offset N` to resume a run that stopped partway. The summaries in the bundled database came from GPT-4o and were reviewed by hand.

## Web app

The Flask app in `visualization/BioiThackCellMAP/` has three tabs:

- **Dashboard:** cohort-level summaries.
- **LLM Interpretation:** the per-patient summaries.
- **NeST-STRING Explorer:** the NeST map with STRING protein interactions and each patient's altered genes.

It reads the reference run and the interpretation database straight from this repo.

```bash
cd visualization/BioiThackCellMAP
pip install flask flask-cors requests
python server.py   # http://localhost:5001
```

## Citation

If you use this code, please cite:

1. Park, S., Silva, E., Singhal, A. et al. A deep learning model of tumor cell architecture elucidates response and resistance to CDK4/6 inhibitors. *Nat Cancer* (2024). https://doi.org/10.1038/s43018-024-00740-1
2. Zhao, Singhal, et al. Cancer mutations converge on a collection of protein assemblies to predict resistance to replication stress. *Cancer Discov* 14(3):508–523 (2024). https://doi.org/10.1158/2159-8290.CD-23-0641
