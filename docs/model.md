# Model and code reference

This page covers how the network is built, how it is trained, the training flags, the code layout and the input file formats. For what the outputs mean, see [outputs.md](outputs.md).

## How the network is built

`DrugCellNN` in `src/drugcell_nn.py` builds its layers from the ontology at startup. The hierarchy has two kinds of edges: system-to-system edges and gene-to-system annotations.

**Gene layer.** Each gene has one to four binary inputs: mutation, copy-number deletion, copy-number amplification and, optionally, fusion. A small per-gene layer, `Linear(n_features → 1) → tanh → BatchNorm1d`, turns them into one number per gene. Batch normalization keeps one feature type from dominating the others.

**System layers.** Systems are built from the leaves up to the root. Each system takes two inputs:
- the hidden states of its child systems
- the outputs of its direct gene layer

These go through `Dropout → Linear → tanh → BatchNorm` to produce a hidden state of size `-genotype_hiddens` (default 4).

**Ontology constraint.** Each system's direct gene layer is a dense `Linear` over all genes, but only its annotated genes are allowed to have weight. At startup, weights outside each system's gene set are zeroed. After every backward pass, their gradients are zeroed too:

```python
param.grad.data = torch.mul(param.grad.data, term_mask_map[term_name])
```

So a gene can only feed a system it is annotated to. This is what makes a system's activity interpretable as the state of its genes. The masking happens in `vnn_trainer.py`; the weight matrices themselves are not sparse. All weights are also scaled by 0.1 at startup.

**Auxiliary outputs.** Every system has a small output head, `Linear(h → 1) → tanh → Linear(1 → 1)`, trained against the same label as the final output:

```
total_loss = main_loss + alpha * sum(aux_loss over all systems)
```

`alpha` defaults to 0.3. Without these heads, systems far from the root get little gradient and learn little.

**Output.** The root's hidden state goes through `Linear(h → 1) → tanh → Linear(1 → 1)` to give the prediction. The forward pass returns two maps:
- the auxiliary outputs, keyed by system name, with the final prediction under `'final'`
- the hidden states of every system and gene

### Genotype only

```mermaid
flowchart TD
    subgraph INPUTS["Genomic Inputs"]
        direction LR
        MUT["Mutations\n(S × genes)"]
        CND["CN Deletions\n(S × genes)"]
        CNA["CN Amplifications\n(S × genes)"]
    end

    DSTACK["np.dstack → Cell Features\n(S × genes × 3)"]
    PERGENE["Per-Gene Feature Layer\nLinear(3 → 1) → tanh → BatchNorm1d(1)\n<i>one per gene</i>"]
    GENEEMB["Gene Embeddings\nscalar per gene → cat → gene_input"]
    DIRECT["Direct Gene Layers\nLinear(num_genes → |annotated_genes|) per term\n<i>weights masked by ontology</i>"]

    subgraph ONTLAYER["Ontology Term Layers (leaf → root)"]
        direction LR
        ONTOLOGY["For each term, bottom-up:\ncat(children_hidden + direct_genes)\n→ Dropout → Linear(in, h) → tanh → BatchNorm(h)"]
        AUX["Aux Head (per term)\nLinear(h → 1) → tanh → Linear(1 → 1)\n<i>supervised against same label</i>"]
    end

    ROOT["Root Hidden Embedding\n(num_hiddens_genotype dims)"]
    FINAL1["Linear(h → 1) → tanh\n<i>final_aux_linear_layer</i>"]
    FINAL2["Linear(1 → 1)\n<i>final_linear_layer_output</i>"]
    PRED(["Prediction\nlogits (binary) or z-score (continuous)"])

    MUT --> DSTACK
    CND --> DSTACK
    CNA --> DSTACK
    DSTACK --> PERGENE
    PERGENE --> GENEEMB
    GENEEMB --> DIRECT
    DIRECT --> ONTOLOGY
    ONTOLOGY -.->|each term| AUX
    ONTOLOGY --> ROOT
    ROOT --> FINAL1
    FINAL1 --> FINAL2
    FINAL2 --> PRED

    style INPUTS fill:#fef3c7,stroke:#f59e0b,color:#1a1a2e
    style MUT fill:#fef3c7,stroke:#f59e0b,color:#1a1a2e
    style CND fill:#fef3c7,stroke:#f59e0b,color:#1a1a2e
    style CNA fill:#fef3c7,stroke:#f59e0b,color:#1a1a2e
    style DSTACK fill:#dbeafe,stroke:#3b82f6,color:#1a1a2e
    style PERGENE fill:#bfdbfe,stroke:#2563eb,color:#1a1a2e
    style GENEEMB fill:#d1fae5,stroke:#10b981,color:#1a1a2e
    style DIRECT fill:#dbeafe,stroke:#3b82f6,color:#1a1a2e
    style ONTLAYER fill:#ede9fe,stroke:#8b5cf6,color:#1a1a2e
    style ONTOLOGY fill:#bfdbfe,stroke:#2563eb,color:#1a1a2e
    style AUX fill:#ede9fe,stroke:#8b5cf6,color:#1a1a2e
    style ROOT fill:#fee2e2,stroke:#ef4444,color:#1a1a2e
    style FINAL1 fill:#dbeafe,stroke:#3b82f6,color:#1a1a2e
    style FINAL2 fill:#dbeafe,stroke:#3b82f6,color:#1a1a2e
    style PRED fill:#fce7f3,stroke:#ec4899,color:#1a1a2e
```

### With clinical covariates

Covariates are the `cov_*` columns that `cbioport_transform.py` adds to `training_data.txt`; `TrainingDataWrapper` picks them up automatically. A clinical encoder, `Linear(n_cov → h) → tanh → BatchNorm`, brings them to the same size as the root's hidden state. The two are concatenated before the final layers. The ontology layers never see the covariates.

```mermaid
flowchart TD
    subgraph GENOMIC["Genomic Pathway"]
        direction TB
        MUT["Mutations\n(S × genes)"]
        CND["CN Deletions\n(S × genes)"]
        CNA["CN Amplifications\n(S × genes)"]
        DSTACK["np.dstack → Cell Features\n(S × genes × 3)"]
        PERGENE["Per-Gene Feature Layer\nLinear(feature_dim → 1) → tanh → BatchNorm1d(1)\n<i>one per gene</i>"]
        GENEEMB["Gene Embeddings\nscalar per gene → cat → gene_input"]
        DIRECT["Direct Gene Layers\nLinear(num_genes → |annotated_genes|) per term\n<i>weights masked by ontology</i>"]
        ONTOLOGY["Ontology Term Layers (leaf → root)\nFor each term, bottom-up:\ncat(children_hidden + direct_genes)\n→ Dropout → Linear(in, h) → tanh → BatchNorm(h)"]
        AUX["Aux Head (per term)\nLinear(h → 1) → tanh → Linear(1 → 1)\n<i>supervised against same label</i>"]
        ROOT["root_hidden\n(h)"]

        MUT --> DSTACK
        CND --> DSTACK
        CNA --> DSTACK
        DSTACK --> PERGENE
        PERGENE --> GENEEMB
        GENEEMB --> DIRECT
        DIRECT --> ONTOLOGY
        ONTOLOGY -.->|each term| AUX
        ONTOLOGY --> ROOT
    end

    subgraph CLINICAL["Clinical Pathway"]
        direction TB
        CLINFEAT["Clinical Features\ncov_* columns from training_data.txt\nMutation count | z-score | FGA\nSample type | Met site | ...\n(S × clin_dim)"]
        CLINENC["Clinical Encoder\nLinear(clin_dim → h) → tanh → BatchNorm(h)\n<i>projects to same dim as root_hidden</i>"]
        CLINH["clinical_h\n(h)"]

        CLINFEAT --> CLINENC
        CLINENC --> CLINH
    end

    subgraph MERGED["Merged Pathway"]
        direction TB
        CONCAT["torch.cat dim=1\n(2h)"]
        FINAL1["Linear(2h → 1) → tanh\n<i>final_aux_linear_layer</i>"]
        FINAL2["Linear(1 → 1)\n<i>final_linear_layer_output</i>"]
        PRED(["Prediction\nlogits or z-score"])

        CONCAT --> FINAL1
        FINAL1 --> FINAL2
        FINAL2 --> PRED
    end

    ROOT --> CONCAT
    CLINH --> CONCAT

    style GENOMIC fill:#2a2050,stroke:#6b5bce,color:#e0d8ff
    style CLINICAL fill:#0f3a34,stroke:#2d8a7a,color:#c8f0e8
    style MERGED fill:#4a2a0a,stroke:#c0783a,color:#ffe0c0

    style MUT fill:#4a3f8a,stroke:#7c6fca,color:#e0d8ff
    style CND fill:#4a3f8a,stroke:#7c6fca,color:#e0d8ff
    style CNA fill:#4a3f8a,stroke:#7c6fca,color:#e0d8ff
    style DSTACK fill:#4a3f8a,stroke:#7c6fca,color:#e0d8ff
    style PERGENE fill:#3d2e9e,stroke:#6b5bce,color:#fff
    style GENEEMB fill:#1a5c3a,stroke:#10b981,color:#c8f0e8
    style DIRECT fill:#3d2e9e,stroke:#6b5bce,color:#fff
    style ONTOLOGY fill:#3d2e9e,stroke:#6b5bce,color:#fff
    style AUX fill:#5a4a8a,stroke:#8b5cf6,color:#e0d8ff
    style ROOT fill:#3a3a4e,stroke:#6a6a80,color:#ccc

    style CLINFEAT fill:#1a5c52,stroke:#2d8a7a,color:#c8f0e8
    style CLINENC fill:#145c52,stroke:#2d8a7a,color:#c8f0e8
    style CLINH fill:#3a3a4e,stroke:#6a6a80,color:#ccc

    style CONCAT fill:#7a4a1a,stroke:#c0783a,color:#ffe0c0
    style FINAL1 fill:#7a4a1a,stroke:#c0783a,color:#ffe0c0
    style FINAL2 fill:#7a4a1a,stroke:#c0783a,color:#ffe0c0
    style PRED fill:#7a4a1a,stroke:#c0783a,color:#ffe0c0
```

| Step | Layer | Input | Output |
|---|---|---|---|
| 1 | `np.dstack` of the feature matrices | 3–4 × (S × genes) | S × genes × features |
| 2 | Per-gene `Linear → tanh → BatchNorm` | S × features, per gene | S × 1, per gene |
| 3 | Concatenate gene outputs | | S × genes |
| 4 | Direct gene layer, per system | S × genes | S × annotated genes |
| 5 | System layers, leaves to root | S × (children's h + direct genes) | S × h |
| 5a | Auxiliary head, per system | S × h | S × 1, used only in the loss |
| 6 | Clinical encoder | S × n_cov | S × h |
| 7 | Concatenate root and clinical | S × h, S × h | S × 2h |
| 8 | `Linear(2h → 1) → tanh` | S × 2h | S × 1 |
| 9 | `Linear(1 → 1)` | S × 1 | S × 1 |

S is the batch size and h is `-genotype_hiddens`.

## Binary and continuous tasks

The network is the same for both. Only the loss and the output handling change.

| | Binary (`-task binary`) | Continuous (`-task continuous`) |
|---|---|---|
| Loss (main and auxiliary) | `BCEWithLogitsLoss` | `MSELoss` |
| Label normalization | None, labels stay 0/1 | Set by `-zscore_method`, see below |
| Prediction output | Logits, then sigmoid, then a 0.5 threshold | Normalized predictions, which can be converted back using `std.txt` |
| Reported metric | Accuracy | Pearson correlation |

For continuous labels, `-zscore_method zscore` or `robustz` normalizes them using training-set statistics and saves the parameters to `std.txt`. The default, `auc`, leaves labels as they are, which suits labels already on a common scale. Prediction always reads `std.txt`, so passing `-zscore_method` to `predict.py` has no effect.

The original NeST-VNN used concordance correlation (CCC) as the auxiliary loss for regression. This version uses MSE instead. Early in training the auxiliary heads output nearly constant values, which makes CCC's denominator close to zero and the loss unstable. MSE has no such problem.

## Training

- **Split:** 20% of samples are held out at random for validation; `-seed` makes the split repeatable. There is no separate test split. `run.py` runs prediction on the whole of `training_data.txt`, training samples included.
- **Optimization:** AdamW with gradient clipping (max norm 5.0). Training stops early when validation loss hasn't improved for `-patience` epochs, and the best checkpoint is kept as `model_final.pt`.
- **Hyperparameter search:** `-optimize 2` runs 20 Optuna trials before the final training run.
- **Architecture images:** each run writes `nn_architecture.png` (the actual network, drawn from the ontology) and `nn_schematic.png` (a simplified overview) to the model folder.
- **MLflow:** on by default; `-no_mlflow` turns it off. A run logs the parameters, the per-epoch losses, metric and gradient norm, the best-epoch values, summary metrics for the saved model, confusion matrices for binary tasks, and the model itself with its input and output signature.

### Training flags

| Flag | Default | Meaning |
|---|---|---|
| `-task` | `continuous` | `binary` or `continuous` |
| `-label` | none | Column of `training_data.txt` to predict |
| `-genotype_hiddens` | `4` | Hidden units per system; the annotation step reads it from the hidden files |
| `-optimize` | `1` | `1` trains directly; `2` runs an Optuna search first |
| `-alpha` | `0.3` | Weight of the auxiliary losses |
| `-epoch` | `300` | Maximum epochs (`run.py` suggests 200) |
| `-lr` | `0.001` | AdamW learning rate |
| `-wd` | `0.001` | AdamW weight decay |
| `-patience` | `30` | Epochs without improvement before stopping |
| `-dropout_fraction` | `0.3` | Dropout on system layers |
| `-min_dropout_layer` | `2` | First layer, counting from the leaves, that gets dropout |
| `-zscore_method` | `auc` | Label normalization for training: `auc` (none), `zscore` or `robustz` |
| `-cuda` | `0` | GPU index, or `cpu` |
| `-seed` | none | Seed for the train/validation split |
| `-no_mlflow` | off | Turn off MLflow tracking |

## Code layout

`src/`:

| File | Role |
|---|---|
| `drugcell_nn.py` | The network |
| `training_data_wrapper.py` | Loads the ontology, checks it has one root and is connected, maps genes to systems, stacks the feature matrices and reads the `cov_*` columns |
| `vnn_trainer.py` | Training loop, gradient masking, early stopping, MLflow logging, architecture images |
| `optuna_nn_trainer.py` | Optuna search, built on `VNNTrainer` |
| `train.py` | Command-line entry point for training |
| `predict.py` | Loads a model and writes predictions, hidden states and gradients |
| `annotate_hierarchy.py` | RLIPP, gene scores, subsystem weights, logic gates, per-patient scores and the HTML viewers |
| `rlipp_calculator.py`, `rlipp_helper.py` | Older RLIPP code that groups results by drug, for the GDSC sample data |
| `util.py` | Data loading, label normalization, gene masks and correlation metrics |

`scripts/`:

| File | Role |
|---|---|
| `cbioport_download.py` | Downloads mutations, copy number, fusions and clinical data for a cBioPortal study |
| `cbioport_transform.py` | Builds the model inputs: you choose the endpoint, the gene panel (by mutation frequency, or from an NDEx hierarchy) and the covariates |
| `run.py` | Interactive launcher for all of the above |

When an NDEx hierarchy is used, systems with fewer than 5 panel genes (the default, as in the paper) are removed. Their genes move up to the nearest remaining parent.

## Input files

All of these live in `data/output/<study_id>/nest_vnn_input/`.

| File | Format | Contents |
|---|---|---|
| `ontology.txt` | TSV: `parent child relation` | `relation` is `default` for a system-to-system edge, `gene` for a gene annotation |
| `cell2ind.txt` | TSV: `index sample_id` | Sets the row order of the feature matrices |
| `gene2ind.txt` | TSV: `index gene_symbol` | Sets the column order of the feature matrices |
| `cell2mutation.txt`, `cell2cndeletion.txt`, `cell2cnamplification.txt`, `cell2fusion.txt` | CSV of 0/1 | Samples × genes |
| `training_data.txt` | TSV with header | `cell_line`, one or more label columns, any `cov_*` columns, and `dataset`. The older headerless format is `cell_line smiles auc dataset`. |
| `endpoints.json` | JSON | Whether each label is binary or continuous; `run.py` uses it to pick the task |
| `covariate_stats.json` | JSON | How each `cov_*` column was encoded (mean and std for numeric columns, a value map for categorical ones), so values can be shown in readable form |
| `metadata.json` | JSON | The cBioPortal study and NDEx network the inputs came from |
