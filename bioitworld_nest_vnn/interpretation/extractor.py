"""Extract per-patient NeST-VNN results from MLflow and local output files."""

import json
from pathlib import Path
from urllib.parse import urlparse

import mlflow
import numpy as np
import pandas as pd

DATA_OUTPUT = Path(__file__).parent.parent / "data" / "output"


def _uri_to_path(artifact_uri: str) -> Path:
    """Convert file:// artifact URI to an absolute local Path."""
    parsed = urlparse(artifact_uri)
    # parsed.path on Windows starts with /C:/..., strip leading slash
    raw = parsed.path.lstrip("/")
    return Path(raw)


def _find_annotation_run(study_id: str, label: str):
    """Return (annotation_run, parent_training_run) for the given study/label."""
    client = mlflow.MlflowClient()
    exp = client.get_experiment_by_name("nest_vnn")
    if exp is None:
        raise ValueError("MLflow experiment 'nest_vnn' not found. Run training first.")

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["start_time DESC"],
    )

    for r in runs:
        params = r.data.params
        if params.get("study_id") != study_id or params.get("label") != label:
            continue
        arts = [a.path for a in client.list_artifacts(r.info.run_id)]
        if "annotation" not in arts:
            continue
        parent_id = r.data.tags.get("mlflow.parentRunId")
        parent = client.get_run(parent_id) if parent_id else None
        return r, parent

    raise ValueError(f"No annotation run found for study_id={study_id!r} label={label!r}")


def extract_patients(study_id: str, label: str, limit: int = None) -> list[dict]:
    """Return list of per-patient dicts ready for LLM prompting."""
    ann_run, _train_run = _find_annotation_run(study_id, label)

    ann_dir = _uri_to_path(ann_run.info.artifact_uri) / "annotation"
    out_dir = DATA_OUTPUT / study_id / label
    inp_dir = DATA_OUTPUT / study_id / "nest_vnn_input"

    # ── Load annotation files from MLflow ──────────────────────────────────
    term_imp = pd.read_csv(ann_dir / "patient_term_importance.txt", sep="\t", index_col="cell_id")
    gene_imp = pd.read_csv(ann_dir / "patient_gene_importance.txt", sep="\t", index_col="cell_id")
    rlipp_pop = pd.read_csv(ann_dir / "rlipp_scores.txt", sep="\t").set_index("term")

    rlipp_pt_path = ann_dir / "patient_rlipp.txt"
    rlipp_pt = pd.read_csv(rlipp_pt_path, sep="\t", index_col="cell_id") if rlipp_pt_path.exists() else None

    top_nest_genes_path = ann_dir / "top_subsystem_genes.txt"
    nest_rep_genes: dict[str, list[str]] = {}
    if top_nest_genes_path.exists():
        tsg = pd.read_csv(top_nest_genes_path, sep="\t")
        for term, grp in tsg.groupby("term"):
            nest_rep_genes[term] = grp.sort_values("rank")["gene"].tolist()

    # ── Load files from data/output (not in MLflow) ────────────────────────
    signed_z_path = out_dir / "annotation" / "patient_gene_signed_z.txt"
    gene_dir_df = pd.read_csv(signed_z_path, sep="\t", index_col="cell_id") if signed_z_path.exists() else None

    prob_path = out_dir / "metrics" / "predict_probabilities.txt"
    probs = np.loadtxt(prob_path).flatten() if prob_path.exists() else None

    # ── Covariates ──────────────────────────────────────────────────────────
    cov_stats_path = inp_dir / "covariate_stats.json"
    cov_stats: list[dict] = []
    if cov_stats_path.exists():
        with open(cov_stats_path) as f:
            cov_stats = json.load(f)

    train_path = inp_dir / "training_data.txt"
    if train_path.exists():
        train_df = pd.read_csv(train_path, sep="\t")
        if "cell_line" in train_df.columns:
            train_df = train_df.set_index("cell_line")
    else:
        train_df = pd.DataFrame()

    # ── Alteration matrices ─────────────────────────────────────────────────
    def _load_mat(fname):
        p = inp_dir / fname
        return np.genfromtxt(p, delimiter=",") if p.exists() else None

    mut_mat = _load_mat("cell2mutation.txt")
    del_mat = _load_mat("cell2cndeletion.txt")
    amp_mat = _load_mat("cell2cnamplification.txt")
    fus_mat = _load_mat("cell2fusion.txt")

    # gene index (0-based row in alteration matrices)
    gene2idx: dict[str, int] = {}
    g2i_path = inp_dir / "gene2ind.txt"
    if g2i_path.exists():
        for line in g2i_path.read_text().splitlines():
            parts = line.strip().split("\t")
            if len(parts) == 2:
                gene2idx[parts[1]] = int(parts[0])

    # patient → row index
    cell2row: dict[str, int] = {}
    c2i_path = inp_dir / "cell2ind.txt"
    if c2i_path.exists():
        for line in c2i_path.read_text().splitlines():
            parts = line.strip().split("\t")
            if len(parts) == 2:
                cell2row[parts[1]] = int(parts[0])

    # ── Helpers ─────────────────────────────────────────────────────────────
    def _denorm(pid: str, col: str) -> float | str | None:
        if col not in train_df.columns or pid not in train_df.index:
            return None
        zval = train_df.loc[pid, col]
        stat = next((s for s in cov_stats if s["label_name"] == col), None)
        if stat is None:
            return None
        if stat["mode"] == "numeric":
            return round(float(zval) * stat["std"] + stat["mean"], 3)
        mapping = stat.get("mapping", {})
        for k, v in mapping.items():
            if abs(float(v) - float(zval)) < 1e-6:
                return k
        return "Primary"

    def _alteration(pid: str, gene: str) -> list[str]:
        row = cell2row.get(pid)
        col = gene2idx.get(gene)
        if row is None or col is None:
            return []
        alts = []
        for mat, tag in [(mut_mat, "mut"), (del_mat, "del"), (amp_mat, "amp"), (fus_mat, "fusion")]:
            if mat is not None and row < mat.shape[0] and col < mat.shape[1] and mat[row, col] == 1:
                alts.append(tag)
        return alts

    # ── Build patient records ───────────────────────────────────────────────
    patient_ids = list(term_imp.index)
    if limit:
        patient_ids = patient_ids[:limit]

    records: list[dict] = []
    for i, pid in enumerate(patient_ids):
        prob = float(probs[i]) if probs is not None and i < len(probs) else 0.5
        pred_class = "high_risk" if prob >= 0.5 else "low_risk"

        # Top 3 NeSTs by absolute importance score
        pt_terms = term_imp.loc[pid].dropna()
        top3_nest_ids = pt_terms.abs().nlargest(3).index.tolist()

        nest_records = []
        for rank, nest_id in enumerate(top3_nest_ids, 1):
            importance = float(term_imp.loc[pid, nest_id])
            direction = "worsened" if importance > 0 else "improved"

            rlipp_score = None
            if rlipp_pt is not None and nest_id in rlipp_pt.columns and pid in rlipp_pt.index:
                rlipp_score = float(rlipp_pt.loc[pid, nest_id])
            pop_rlipp = float(rlipp_pop.loc[nest_id, "rlipp"]) if nest_id in rlipp_pop.index else None
            pop_prho = float(rlipp_pop.loc[nest_id, "p_rho"]) if nest_id in rlipp_pop.index else None

            rep_genes = nest_rep_genes.get(nest_id, [])[:5]

            # Top 3 patient genes within this NEST's gene set
            candidate_genes = [g for g in gene_imp.columns if g in set(rep_genes)] or list(gene_imp.columns)
            pt_gene_scores = gene_imp.loc[pid, candidate_genes].dropna()
            top3_genes = pt_gene_scores.abs().nlargest(3).index.tolist()

            gene_records = []
            for grank, gene in enumerate(top3_genes, 1):
                g_imp = float(gene_imp.loc[pid, gene])
                g_dir = "improved"
                if gene_dir_df is not None and gene in gene_dir_df.columns and pid in gene_dir_df.index:
                    g_dir = "worsened" if gene_dir_df.loc[pid, gene] > 0 else "improved"
                alts = _alteration(pid, gene)
                gene_records.append({
                    "gene_name": gene,
                    "rank": grank,
                    "importance_score": round(g_imp, 4),
                    "direction": g_dir,
                    "alteration_types": alts or ["not_altered"],
                })

            nest_records.append({
                "nest_id": nest_id,
                "rank": rank,
                "importance_score": round(importance, 4),
                "rlipp_score": round(rlipp_score, 4) if rlipp_score is not None else None,
                "population_rlipp": round(pop_rlipp, 4) if pop_rlipp is not None else None,
                "population_p_rho": round(pop_prho, 4) if pop_prho is not None else None,
                "direction": direction,
                "representative_genes": rep_genes,
                "top_patient_genes": gene_records,
            })

        records.append({
            "patient_id": pid,
            "study_id": study_id,
            "label": label,
            "predicted_probability": round(prob, 4),
            "predicted_class": pred_class,
            "sample_type": _denorm(pid, "cov_sample_type") or "Unknown",
            "mutation_count": _denorm(pid, "cov_mutation_count"),
            "fraction_genome_altered": _denorm(pid, "cov_fraction_genome_altered"),
            "top_nests": nest_records,
        })

    return records
