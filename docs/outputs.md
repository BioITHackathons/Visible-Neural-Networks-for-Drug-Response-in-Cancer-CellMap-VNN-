# Reading NeST-VNN outputs

A run writes everything under `data/output/<study_id>/<label>/`. This page lists those files and explains the scores in them.

One caveat applies to every score here. RLIPP, gene scores and subsystem weights are all computed against the model's own predictions, not the true labels. They describe what the model has learned to rely on. If the model predicts poorly, the scores describe a poor model, so check its accuracy before drawing biological conclusions.

## Files

### `model/`

| File | Contents |
|---|---|
| `model_final.pt` | Checkpoint with the lowest validation loss |
| `std.txt` | Label normalization parameters, reused at prediction time |

### `metrics/`

| File | Contents |
|---|---|
| `predict.txt` | Raw output: logits for binary tasks, z-scores for continuous ones |
| `predict_probabilities.txt` | Sigmoid probabilities (binary only) |
| `predict_predictions.txt` | Predictions thresholded at 0.5 (binary only) |
| `hidden/<term>.hidden` | Hidden state of each NeST system, samples × hidden units |
| `hidden/<gene>.hidden` | Hidden state of each gene, samples × 1 |

### `annotation/`

Cohort-level scores:

| File | Contents |
|---|---|
| `rlipp_scores.txt` | RLIPP score for each system |
| `top_systems.txt` | The 20 systems with the highest RLIPP |
| `gene_scores.txt` | Correlation of each gene with the prediction |
| `subsystem_gene_weights.txt` | For each system, its genes ranked by how much they drive it |
| `top_subsystem_genes.txt` | Top 5 driving genes for each of the 20 highest-RLIPP systems |
| `boolean_logic.txt` | Best-fitting logic gate for pairs of child systems |

Per-patient scores (each a samples × terms or samples × genes table):

| File | Contents |
|---|---|
| `patient_term_importance.txt` | How unusual each system's activity is for that patient |
| `patient_term_mean_z.txt` | Signed version of the above, used to show direction |
| `patient_rlipp.txt` | Patient-level RLIPP |
| `patient_gene_importance.txt` | Absolute z-score of each gene's hidden state |
| `patient_gene_signed_z.txt` | Signed z-score of each gene's hidden state |

Viewers and graph exports:

| File | Contents |
|---|---|
| `hierarchy_viz.html` | Browsable hierarchy with all cohort scores; opens in any browser |
| `patient_viz.html` | Per-patient viewer; also the data source for the web app |
| `hierarchy_annotated.graphml` | Annotated hierarchy for Cytoscape |
| `hierarchy_annotated.cx2` | The same for NDEx and Cytoscape Web (needs `ndex2`) |

## RLIPP scores (`rlipp_scores.txt`)

RLIPP asks whether a system adds predictive signal beyond what its child systems already carry.

For each system, the annotation step reduces its hidden state with PCA, fits a ridge regression against the model's predictions, and takes the Spearman correlation of the fit (`p_rho`). It then does the same with the combined hidden states of the system's children (`c_rho`). RLIPP is `p_rho / c_rho`. It is set to 0 when `c_rho` is close to zero.

| Column | Meaning |
|---|---|
| `term` | System ID, e.g. `NEST:38` |
| `rlipp` | `p_rho / c_rho` |
| `p_rho`, `p_pval` | Correlation of the system's own hidden state with the prediction, and its p-value |
| `c_rho`, `c_pval` | The same for its children combined |

How to read it:

- **Above 1:** the system carries signal its children don't. These are the systems the model leans on at that level of the hierarchy.
- **About 1:** the children already explain it.
- **Below 1:** the children predict better than the parent. The parent may just be summarizing them.

A high `p_rho` with a small `p_pval` means the system's activity tracks the prediction reliably across patients.

## Gene scores (`gene_scores.txt`)

Each gene's hidden state is a single number per patient. `rho` is the Spearman correlation of that number with the model's predictions, and `p_val` is its p-value. A positive `rho` means higher activation goes with a higher predicted outcome. Rows are sorted by `|rho|`.

## Subsystem gene weights (`subsystem_gene_weights.txt`)

For each system, this file ranks the genes annotated to it. It does this by correlating each gene's hidden state with the first principal component (PC1) of the system's hidden state.

| Column | Meaning |
|---|---|
| `term`, `gene` | System and gene |
| `pc1_corr`, `pc1_corr_pval` | Spearman correlation with the system's PC1, and its p-value |
| `rank` | Rank within the system by `|pc1_corr|` (1 = strongest) |

The sign shows whether the gene pushes the system along its main direction of variation or against it. `top_subsystem_genes.txt` lists the top 5 genes for the 20 highest-RLIPP systems. The same genes appear as "Key driving genes" in `hierarchy_viz.html`.

## Boolean logic (`boolean_logic.txt`)

This file describes how a parent system combines two of its child systems, following Ma et al. 2018 (DCell).

For every parent with at least two child systems, each pair of children (A, B) is tested:

1. Each system's activity is set to on or off, depending on whether its PC1 is above the cohort median.
2. For each of the four (A, B) combinations, the parent's majority state is recorded.
3. That truth table is matched against the 10 non-trivial two-input gates below.

A pair is only scored if each of the four combinations covers at least 4 samples and no more than half of all samples.

| Column | Meaning |
|---|---|
| `parent`, `child1`, `child2` | The three systems |
| `logic` | Best-matching gate |
| `consistency` | Share of samples where the gate predicts the parent's state correctly |
| `n_samples` | Samples used |

| Gate | Parent is on when | Reading |
|---|---|---|
| `AND` | both A and B are on | both are required |
| `OR` | A or B is on | either one is enough |
| `XOR` | exactly one is on | they are mutually exclusive |
| `NAND` | not both are on | turning both on shuts the parent off |
| `NOR` | neither is on | either one shuts the parent off |
| `XNOR` | both are on, or both are off | the parent follows their agreement |
| `A_NOT_B` | A is on and B is off | A drives it, B inhibits it |
| `B_NOT_A` | B is on and A is off | B drives it, A inhibits it |
| `A_OR_NOT_B` | any state except B on alone | |
| `B_OR_NOT_A` | any state except A on alone | |

Treat consistency of 0.70 or more as a good fit, 0.60–0.70 as moderate, and below 0.60 as noise. The gates describe patterns in the model's learned representations, not biochemical mechanisms. They are most meaningful for high-RLIPP systems. They also depend on the median cutoff used in step 1.

## Hierarchy viewer (`hierarchy_viz.html`)

Open the file in a browser; it needs no server. Pick a system from the table to see:

- its RLIPP, `p_rho`, `p_pval`, `c_rho` and `c_pval`
- its parent and child systems, which are clickable
- its key driving genes, from `subsystem_gene_weights.txt`
- its logic-gate fits with its children, colored by consistency, with an in-page explanation under **? help**
- genes annotated directly to it, and genes it inherits from its children, each with its gene-level `rho`

The table can be sorted, filtered and searched.

## Patient viewer (`patient_viz.html`)

Pick a patient by ID. The page has three parts.

### Summary bar

- **Prediction:** for binary tasks, the predicted probability is labeled high (≥ 0.7), moderate-high (≥ 0.5), moderate-low (≥ 0.3) or low. For continuous tasks, the score is shown relative to the cohort mean.
- **Top system:** the patient's most important system and its cohort RLIPP. It is marked "cohort-validated" if that RLIPP is above 1.2.
- **Top gene:** the most important gene with a clear direction (|z| ≥ 0.8 and |cohort `rho`| ≥ 0.1). For example: "TP53 (z=+2.1, cohort ρ=+0.41) → higher".

### Systems panel

Every system, ranked by importance for this patient:

| Column | Meaning |
|---|---|
| Importance | Length of the patient's z-scored hidden state for the system, i.e. how far it sits from the cohort average. Raw activations are nearly the same for everyone because of batch normalization, so z-scoring is needed to see differences. |
| Pt-RLIPP | The system's squared deviation divided by the sum of its children's. Above 1 means the system is more unusual than its children explain. |
| Pop-RLIPP | The cohort RLIPP from `rlipp_scores.txt` |
| Outcome Dir | sign(mean z across hidden units) × sign(cohort `p_rho`). Shown only when \|mean z\| ≥ 0.2 and \|`p_rho`\| ≥ 0.1. |

### Genes panel

The patient's 100 most important genes:

| Column | Meaning |
|---|---|
| Importance | \|z-score\| of the gene's hidden state |
| Signed Z | The same z-score with its sign: above or below the cohort average |
| Outcome Dir | sign(signed z) × sign(cohort `rho`). "↑ higher" (orange) means the deviation points toward a higher predicted score; "↓ lower" (green) means the opposite. Shown only when \|z\| ≥ 0.5 and \|cohort `rho`\| ≥ 0.1, otherwise "—". |

Both direction columns are rough guides, not causal claims. They combine a cohort-wide correlation with one patient's deviation and assume the relationship is monotone. The system version also reduces a multi-dimensional state to its mean. Projecting onto the ridge regression direction would be more principled. Use them to generate hypotheses.
