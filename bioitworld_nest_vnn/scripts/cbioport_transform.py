"""
Transform cBioPortal output into NeST-VNN input files.

Interactive CLI lets you pick which clinical attributes to use as endpoints.
Works with any cBioPortal study downloaded by cbioportal_download.py.

Usage:
    python transform_to_nest_vnn.py <cbioportal_dir>
    python transform_to_nest_vnn.py cbioportal_laml_tcga_pub
    python transform_to_nest_vnn.py cbioportal_breast_msk_2025

The script will:
  1. Load the downloaded CSVs
  2. Show all available clinical attributes
  3. Let you interactively pick which ones to use as label columns
  4. For status/categorical columns, show unique values and let you pick 0/1 mapping
  5. Generate NeST-VNN input files with a single training_data.txt
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import shutil
import json
import requests
from datetime import datetime


# ── Configuration ────────────────────────────────────────────────────────────

NEST_VNN_SAMPLE = Path("nest_vnn/sample")

CNV_DEEP_DELETION = -2
CNV_AMPLIFICATION = 2

DEFAULT_NDEX_UUID = "9a8f5326-aa6e-11ea-aaef-0ac135e8bacf"


# ── CLI helpers ──────────────────────────────────────────────────────────────

def prompt_choice(prompt: str, options: list[str], allow_multi: bool = False) -> list[int] | int:
    """Display numbered options and get user selection."""
    print(f"\n{prompt}")
    for i, opt in enumerate(options):
        print(f"  [{i}] {opt}")

    if allow_multi:
        print(f"\nEnter numbers separated by commas (e.g. 0,3,5), or 'done' to finish:")
    else:
        print(f"\nEnter number:")

    while True:
        raw = input("> ").strip()
        if raw.lower() in ('done', 'q', 'quit', 'skip', 'none', ''):
            return []
        try:
            if allow_multi:
                indices = [int(x.strip()) for x in raw.split(",")]
            else:
                indices = [int(raw)]
            if all(0 <= i < len(options) for i in indices):
                return indices if allow_multi else indices[0]
        except ValueError:
            pass
        print(f"Invalid input. Enter number(s) 0-{len(options)-1}.")


def prompt_binary_mapping(col_name: str, unique_values: list[str]) -> dict[str, float] | None:
    """Let user pick which values map to 0 (event) and 1 (no event)."""
    print(f"\n  Column '{col_name}' has these unique values:")
    for i, v in enumerate(unique_values):
        print(f"    [{i}] {v}")

    print(f"\n  Which values should map to 0 (event / bad outcome)?")
    print(f"  Enter numbers separated by commas:")
    while True:
        raw = input("  event(0)> ").strip()
        if raw.lower() in ('skip', 'q', ''):
            return None
        try:
            event_indices = [int(x.strip()) for x in raw.split(",")]
            if all(0 <= i < len(unique_values) for i in event_indices):
                break
        except ValueError:
            pass
        print(f"  Invalid. Enter number(s) 0-{len(unique_values)-1}, or 'skip'.")

    print(f"\n  Which values should map to 1 (no event / good outcome)?")
    print(f"  Enter numbers separated by commas:")
    while True:
        raw = input("  no-event(1)> ").strip()
        if raw.lower() in ('skip', 'q', ''):
            return None
        try:
            noevent_indices = [int(x.strip()) for x in raw.split(",")]
            if all(0 <= i < len(unique_values) for i in noevent_indices):
                break
        except ValueError:
            pass
        print(f"  Invalid. Enter number(s) 0-{len(unique_values)-1}, or 'skip'.")

    mapping = {}
    for i in event_indices:
        mapping[unique_values[i]] = 0.0
    for i in noevent_indices:
        mapping[unique_values[i]] = 1.0

    print(f"  Mapping: { {k: int(v) for k, v in mapping.items()} }")
    return mapping


def classify_column(series: pd.Series) -> str:
    """Classify a clinical column as 'numeric', 'binary_candidate', or 'categorical'."""
    non_null = series.dropna()
    if non_null.empty:
        return "empty"
    try:
        pd.to_numeric(non_null)
        return "numeric"
    except (ValueError, TypeError):
        pass
    n_unique = non_null.nunique()
    if n_unique <= 10:
        return "binary_candidate"
    return "categorical"


def interactive_endpoint_selection(clinical_df: pd.DataFrame) -> list[dict]:
    """
    Interactive CLI to select clinical attributes as label columns.
    Returns list of endpoint dicts:
        {"label_name": str, "clinical_col": str, "mode": "continuous"|"binary",
         "mapping": dict|None}
    """
    if clinical_df.empty:
        print("⚠ No clinical data available for endpoint selection.")
        return []

    # Skip ID columns
    skip_cols = {"patientId", "sampleId", "PATIENT_ID", "SAMPLE_ID",
                 "studyId", "uniquePatientKey", "uniqueSampleKey"}
    available = [c for c in clinical_df.columns if c not in skip_cols]

    # Classify and summarize each column
    col_info = []
    for col in available:
        series = clinical_df[col].dropna()
        if series.empty:
            continue
        col_type = classify_column(series)
        if col_type == "empty":
            continue

        if col_type == "numeric":
            vals = pd.to_numeric(series)
            summary = f"numeric, n={len(vals)}, median={vals.median():.1f}, range=[{vals.min():.1f}, {vals.max():.1f}]"
        elif col_type == "binary_candidate":
            counts = series.value_counts()
            summary = f"categorical ({series.nunique()} values): " + ", ".join(
                f"{v}({c})" for v, c in counts.items()
            )
        else:
            summary = f"categorical ({series.nunique()} unique values), n={len(series)}"

        col_info.append((col, col_type, summary))

    # Display
    print("\n" + "=" * 70)
    print("AVAILABLE CLINICAL ATTRIBUTES")
    print("=" * 70)
    options = []
    for i, (col, ctype, summary) in enumerate(col_info):
        label = "📊" if ctype == "numeric" else "🏷️"
        print(f"  [{i:2d}] {label} {col:40s} {summary}")
        options.append(col)

    print("\n" + "-" * 70)
    print("Select which attributes to use as endpoints/labels.")
    print("You can add multiple. Enter numbers separated by commas, or 'done' when finished.")
    print("-" * 70)

    indices = prompt_choice(
        "Which clinical attributes should be endpoints?",
        [f"{col} — {summary}" for col, _, summary in col_info],
        allow_multi=True,
    )

    if not indices:
        print("No endpoints selected.")
        return []

    endpoints = []
    for idx in indices:
        col, col_type, _ = col_info[idx]
        series = clinical_df[col].dropna()

        if col_type == "numeric":
            # Offer both continuous and binary (thresholded)
            print(f"\n  '{col}' is numeric. Use as:")
            mode_idx = prompt_choice(
                f"  How to use '{col}'?",
                ["Continuous (regression)", "Binary (threshold)", "Both"],
            )
            if isinstance(mode_idx, list) and not mode_idx:
                continue

            if mode_idx in (0, 2):
                label_name = col.lower().replace(" ", "_").replace("-", "_")
                endpoints.append({
                    "label_name": label_name,
                    "clinical_col": col,
                    "mode": "continuous",
                    "mapping": None,
                })

            if mode_idx in (1, 2):
                vals = pd.to_numeric(series)
                print(f"    median={vals.median():.1f}, mean={vals.mean():.1f}")
                thresh = input(f"    Enter threshold (values > threshold = 1): ").strip()
                try:
                    thresh = float(thresh)
                    label_name = f"binary_{col.lower().replace(' ', '_').replace('-', '_')}"
                    endpoints.append({
                        "label_name": label_name,
                        "clinical_col": col,
                        "mode": "binary_threshold",
                        "mapping": {"threshold": thresh},
                    })
                except ValueError:
                    print("    Invalid threshold, skipping binary version.")

        elif col_type == "binary_candidate":
            unique_vals = sorted(series.unique().tolist(), key=str)
            mapping = prompt_binary_mapping(col, unique_vals)
            if mapping:
                label_name = f"binary_{col.lower().replace(' ', '_').replace('-', '_')}"
                endpoints.append({
                    "label_name": label_name,
                    "clinical_col": col,
                    "mode": "binary_status",
                    "mapping": mapping,
                })
        else:
            print(f"  '{col}' has {series.nunique()} unique values — too many for binary.")
            print(f"  Skipping (use a column with fewer categories).")

    # Summary
    print(f"\n{'─'*70}")
    print(f"Selected {len(endpoints)} endpoint(s):")
    for ep in endpoints:
        print(f"  {ep['label_name']:30s} ← {ep['clinical_col']} ({ep['mode']})")
    print(f"{'─'*70}")

    return endpoints


# ── Covariate selection ───────────────────────────────────────────────────────

def interactive_covariate_selection(clinical_df: pd.DataFrame, endpoint_cols: set) -> list[dict]:
    """
    Interactive CLI to select clinical covariates to feed into the model.
    Returns list of covariate dicts with encoding parameters (cov_* label names).
    """
    if clinical_df.empty:
        return []

    skip_cols = {"patientId", "sampleId", "PATIENT_ID", "SAMPLE_ID",
                 "studyId", "uniquePatientKey", "uniqueSampleKey", "dataset"}
    skip_cols |= set(endpoint_cols)

    RECOMMENDED = {"SAMPLE_TYPE", "MUTATION_COUNT", "FRACTION_GENOME_ALTERED", "TUMOR_PURITY"}

    col_info = []
    for col in clinical_df.columns:
        if col in skip_cols:
            continue
        series = clinical_df[col].dropna()
        if series.empty:
            continue
        if 1.0 - len(series) / len(clinical_df) > 0.6:
            continue
        col_type = classify_column(series)
        if col_type in ("empty", "categorical"):
            continue
        if col_type == "numeric":
            vals = pd.to_numeric(series, errors='coerce').dropna()
            summary = f"numeric   n={len(vals)}  median={vals.median():.1f}"
        else:
            counts = series.value_counts()
            summary = "binary    " + "  /  ".join(f"{v}({c})" for v, c in counts.head(4).items())
        star = "★ " if col in RECOMMENDED else "  "
        col_info.append((col, col_type, summary, star))

    if not col_info:
        return []

    print("\n" + "=" * 70)
    print("COVARIATE FEATURES (optional model inputs)")
    print("=" * 70)
    print("Covariates are sample-level clinical features fed alongside genomic")
    print("data into the final prediction layer. ★ = recommended.")
    print()
    for i, (col, _, summary, star) in enumerate(col_info):
        print(f"  [{i:2d}] {star}{col:38s} {summary}")

    print("\nEnter numbers to include as covariates (e.g. 0,2,3), or press Enter to skip:")
    indices = prompt_choice("", [f"{star}{col}" for col, _, _, star in col_info], allow_multi=True)

    if not indices:
        print("No covariates selected — genomic features only.")
        return []

    SMART_DEFAULTS = {
        "SAMPLE_TYPE":    {"Metastasis": 1.0},
        "OS_STATUS":      {"1:DECEASED": 1.0},
        "SOMATIC_STATUS": {"Matched": 1.0},
    }

    covariates = []
    for idx in indices:
        col, col_type, _, _ = col_info[idx]
        series = clinical_df[col].dropna()
        label_name = f"cov_{col.lower().replace(' ', '_').replace('-', '_')}"

        if col_type == "numeric":
            vals = pd.to_numeric(series, errors='coerce').dropna()
            median_val = float(vals.median())
            mean_val = float(vals.mean())
            std_val = float(vals.std())
            if std_val == 0.0 or np.isnan(std_val):
                std_val = 1.0
            covariates.append({
                "clinical_col": col,
                "label_name": label_name,
                "mode": "numeric",
                "median": median_val,
                "mean": mean_val,
                "std": std_val,
            })
            print(f"  ✓ {col:35s} → {label_name}  (z-score: μ={mean_val:.2f}, σ={std_val:.2f})")

        else:
            unique_vals = sorted(series.unique().tolist(), key=str)
            default_mapping = SMART_DEFAULTS.get(col)
            used_default = False
            if default_mapping:
                present = {str(v) for v in unique_vals}
                if all(str(k) in present for k in default_mapping):
                    print(f"\n  '{col}' suggested encoding: {default_mapping}")
                    confirm = input(f"  Use this mapping? [Y/n]: ").strip().lower()
                    if confirm in ('', 'y', 'yes'):
                        covariates.append({
                            "clinical_col": col,
                            "label_name": label_name,
                            "mode": "binary",
                            "mapping": {str(k): v for k, v in default_mapping.items()},
                            "default_value": 0.0,
                        })
                        print(f"  ✓ {col:35s} → {label_name}")
                        used_default = True
            if not used_default:
                mapping = prompt_binary_mapping(col, [str(v) for v in unique_vals])
                if mapping:
                    covariates.append({
                        "clinical_col": col,
                        "label_name": label_name,
                        "mode": "binary",
                        "mapping": mapping,
                        "default_value": 0.0,
                    })
                    print(f"  ✓ {col:35s} → {label_name}")

    print(f"\n{'─'*70}")
    print(f"Selected {len(covariates)} covariate(s):")
    for cov in covariates:
        print(f"  {cov['label_name']:35s} ← {cov['clinical_col']} ({cov['mode']})")
    print(f"{'─'*70}")
    return covariates


def build_covariates(clinical_df: pd.DataFrame, sample_ids: list, covariates: list) -> dict:
    """
    Encode covariate columns for all sample_ids and return {label_name: np.ndarray}.
    Numeric covariates are z-score normalized; binary covariates are 0/1 encoded.
    Missing values are imputed with 0 (which equals the mean for z-scored numerics).
    """
    if not covariates:
        return {}

    clin_by_sample = {}
    clin_by_patient = {}
    if "sampleId" in clinical_df.columns:
        for _, row in clinical_df.iterrows():
            clin_by_sample[row["sampleId"]] = row
    if "patientId" in clinical_df.columns:
        for _, row in clinical_df.iterrows():
            pid = row["patientId"]
            if pid not in clin_by_patient:
                clin_by_patient[pid] = row

    sample_to_patient = {}
    if clin_by_patient:
        patient_ids = sorted(clin_by_patient.keys(), key=len, reverse=True)
        for sid in sample_ids:
            if sid in clin_by_sample:
                continue
            for pid in patient_ids:
                if sid.startswith(pid):
                    sample_to_patient[sid] = pid
                    break

    def resolve(sid):
        if sid in clin_by_sample:
            return clin_by_sample[sid]
        if sid in sample_to_patient:
            return clin_by_patient[sample_to_patient[sid]]
        if sid in clin_by_patient:
            return clin_by_patient[sid]
        return None

    result = {}
    for cov in covariates:
        arr = np.zeros(len(sample_ids))
        for i, sid in enumerate(sample_ids):
            record = resolve(sid)
            if record is None:
                continue
            raw = record.get(cov["clinical_col"], None)
            if raw is None or (isinstance(raw, float) and pd.isna(raw)):
                continue
            if cov["mode"] == "numeric":
                try:
                    val = float(str(raw).strip())
                    arr[i] = (val - cov["mean"]) / cov["std"]
                except (ValueError, TypeError):
                    pass
            else:
                arr[i] = cov["mapping"].get(str(raw).strip(), cov.get("default_value", 0.0))
        result[cov["label_name"]] = arr
    return result


# ── NDEx ontology builder ────────────────────────────────────────────────────

def download_ndex_cx2(uuid: str) -> list:
    """Download a CX2 network from NDEx Public by UUID."""
    url = f"https://www.ndexbio.org/v3/networks/{uuid}"
    resp = requests.get(url, headers={"Accept": "application/json"}, timeout=120)
    resp.raise_for_status()
    return resp.json()


def parse_cx2_hierarchy(cx2_data: list) -> tuple[dict, dict, list]:
    """
    Parse a CX2 document into (node_names, node_genes, edges).

    node_names : {node_id: str}
    node_genes : {node_id: [gene_symbol, ...]}  from CD_MemberList or similar attrs
    edges      : [(source_id, target_id)]        hierarchy edges, parent → child
    """
    node_names: dict[int, str] = {}
    node_genes: dict[int, list] = {}
    edges: list[tuple] = []

    MEMBER_ATTRS = ["Genes", "CD_MemberList", "member", "genes", "HiDeF_persistence"]

    for aspect in cx2_data:
        if not isinstance(aspect, dict):
            continue

        if "nodes" in aspect:
            for node in aspect["nodes"]:
                nid = node["id"]
                attrs = node.get("v", {})
                node_names[nid] = attrs.get("n", attrs.get("name", str(nid)))

                genes: list[str] = []
                for attr in MEMBER_ATTRS:
                    val = attrs.get(attr)
                    if isinstance(val, str) and val.strip():
                        genes = [g.strip() for g in val.split() if g.strip()]
                        break
                    elif isinstance(val, list):
                        genes = [str(g).strip() for g in val if str(g).strip()]
                        break
                node_genes[nid] = genes

        elif "edges" in aspect:
            for edge in aspect["edges"]:
                edges.append((edge["s"], edge["t"]))

    return node_names, node_genes, edges


def compute_altered_genes(mut_df: pd.DataFrame, cnv_df: pd.DataFrame,
                          fusions_df: pd.DataFrame, sample_ids: list,
                          min_freq: float) -> set[str]:
    """Return gene symbols altered in >= min_freq fraction of samples (any data type)."""
    sample_set = set(sample_ids)
    threshold = min_freq * len(sample_ids)
    gene_col_candidates = ["gene.hugoGeneSymbol", "hugoGeneSymbol"]

    pairs: list[pd.DataFrame] = []   # accumulate (gene, sample) frames

    if not mut_df.empty:
        gc = next((c for c in gene_col_candidates if c in mut_df.columns), None)
        if gc and "sampleId" in mut_df.columns:
            df = mut_df[[gc, "sampleId"]].rename(columns={gc: "gene", "sampleId": "sample"})
            df["gene"] = df["gene"].astype(str).str.strip()
            df["sample"] = df["sample"].astype(str)
            pairs.append(df[df["sample"].isin(sample_set) & df["gene"].ne("") & df["gene"].ne("nan")])

    if not cnv_df.empty:
        gc = next((c for c in gene_col_candidates if c in cnv_df.columns), None)
        vc = next((c for c in ["alteration", "value"] if c in cnv_df.columns), None)
        if gc and vc and "sampleId" in cnv_df.columns:
            df = cnv_df[[gc, "sampleId", vc]].copy()
            df[vc] = pd.to_numeric(df[vc], errors="coerce")
            df = df[df[vc].notna() & ((df[vc] <= CNV_DEEP_DELETION) | (df[vc] >= CNV_AMPLIFICATION))]
            df = df.rename(columns={gc: "gene", "sampleId": "sample"})[["gene", "sample"]]
            df["gene"] = df["gene"].astype(str).str.strip()
            df["sample"] = df["sample"].astype(str)
            pairs.append(df[df["sample"].isin(sample_set) & df["gene"].ne("") & df["gene"].ne("nan")])

    if not fusions_df.empty:
        sv_cols = [c for c in ["site1HugoSymbol", "site2HugoSymbol",
                                "site1.hugoSymbol", "site2.hugoSymbol",
                                "gene1.hugoGeneSymbol", "gene2.hugoGeneSymbol"]
                   if c in fusions_df.columns]
        if not sv_cols:
            sv_cols = [c for c in gene_col_candidates if c in fusions_df.columns]
        if sv_cols and "sampleId" in fusions_df.columns:
            df = fusions_df[["sampleId"] + sv_cols].copy()
            df["sampleId"] = df["sampleId"].astype(str)
            df = df[df["sampleId"].isin(sample_set)]
            melted = (df.melt(id_vars="sampleId", value_vars=sv_cols, value_name="gene")
                        .rename(columns={"sampleId": "sample"})[["gene", "sample"]])
            melted["gene"] = melted["gene"].astype(str).str.strip()
            pairs.append(melted[melted["gene"].ne("") & melted["gene"].ne("nan")])

    if not pairs:
        return set()

    counts = (pd.concat(pairs, ignore_index=True)
                .drop_duplicates()
                .groupby("gene")["sample"]
                .nunique())
    return set(counts[counts >= threshold].index)


def build_ontology_from_ndex(uuid: str, gene_set: set, min_genes: int = 5) -> tuple[dict, list]:
    """
    Download hierarchy from NDEx and build gene2ind + ontology rows filtered to gene_set.

    Only assemblies whose CD_MemberList contains >= min_genes panel genes are kept
    (matching the NeST-VNN paper: "assemblies encoded by at least five genes represented
    on the 718-gene clinical panel"). Pruned intermediate nodes are bridged over so their
    surviving children are promoted to the nearest surviving ancestor.

    Returns:
        gene2ind     : {gene_symbol: index}  sorted alphabetically
        ontology_rows: [(parent, child, relation)]  ready to write as ontology.txt
    """
    print(f"\nDownloading NDEx network {uuid} ...")
    cx2_data = download_ndex_cx2(uuid)
    node_names, node_genes, raw_edges = parse_cx2_hierarchy(cx2_data)
    all_network_genes = {g for genes in node_genes.values() for g in genes}
    print(f"  {len(node_names)} nodes, {len(raw_edges)} edges, "
          f"{len(all_network_genes)} unique gene members in network")

    # Filter gene members to the requested gene_set
    filtered: dict[int, set] = {
        nid: {g for g in genes if g in gene_set}
        for nid, genes in node_genes.items()
    }

    # Build parent/children maps
    children_of: dict[int, list] = {}
    parent_of: dict[int, list] = {}
    for src, tgt in raw_edges:
        children_of.setdefault(src, []).append(tgt)
        parent_of.setdefault(tgt, []).append(src)

    # Keep only terms whose CD_MemberList (= full HiDeF cluster membership, already
    # includes all sub-cluster genes) has >= min_genes panel genes.
    surviving = {nid for nid in node_names if len(filtered.get(nid, set())) >= min_genes}
    n_pruned = len(node_names) - len(surviving)
    if n_pruned:
        print(f"  Pruned {n_pruned} assemblies with <{min_genes} panel genes "
              f"(paper threshold: at least {min_genes})")

    # For hierarchy edges, bridge over pruned intermediate nodes: find the nearest
    # surviving ancestor for each surviving node, then emit a direct edge.
    def surviving_parents(nid: int, visited: set | None = None) -> set:
        if visited is None:
            visited = set()
        if nid in visited:
            return set()
        visited.add(nid)
        result = set()
        for p in parent_of.get(nid, []):
            if p in surviving:
                result.add(p)
            else:
                result |= surviving_parents(p, visited)
        return result

    # Deduplicate gene annotations: CD_MemberList in HiDeF hierarchies includes
    # all descendant members at every level, so a gene that belongs to a child term
    # will also appear in every ancestor's member list. After pruning, keep only the
    # most specific (deepest surviving) assignment for each gene.
    direct_genes: dict[int, set] = {nid: set(filtered.get(nid, set())) for nid in surviving}

    def descendant_genes_surviving(nid: int, visited: set | None = None) -> set:
        if visited is None:
            visited = set()
        if nid in visited:
            return set()
        visited.add(nid)
        result = set()
        for child in children_of.get(nid, []):
            if child in surviving:
                result |= direct_genes.get(child, set())
                result |= descendant_genes_surviving(child, visited)
            else:
                # skip pruned node but continue through its children
                result |= descendant_genes_surviving(child, visited)
        return result

    deduped: dict[int, set] = {}
    for nid in surviving:
        deduped[nid] = direct_genes[nid] - descendant_genes_surviving(nid)

    all_annotated = {g for genes in deduped.values() for g in genes}
    print(f"  Input gene set: {len(gene_set)} genes")
    print(f"  Genes annotated in ontology: {len(all_annotated)}")
    print(f"  Surviving terms: {len(surviving)} / {len(node_names)}")
    n_removed = sum(len(direct_genes[n]) - len(deduped[n]) for n in surviving)
    if n_removed:
        print(f"  Removed {n_removed} redundant parent gene annotations (genes already in a child term)")

    if not all_annotated:
        raise ValueError(
            "No genes from the selected set are annotated in this NDEx hierarchy. "
            "Check that gene symbols match (HGNC format expected)."
        )

    gene2ind = {g: i for i, g in enumerate(sorted(all_annotated))}

    ontology_rows: list[tuple] = []
    for nid in surviving:
        for par in surviving_parents(nid):
            ontology_rows.append((node_names[par], node_names[nid], "default"))
    for nid in surviving:
        for gene in deduped[nid]:
            ontology_rows.append((node_names[nid], gene, "gene"))

    n_term_edges = sum(1 for r in ontology_rows if r[2] == "default")
    n_gene_edges = sum(1 for r in ontology_rows if r[2] == "gene")
    print(f"  Ontology: {len(surviving)} terms, "
          f"{n_term_edges} hierarchy edges, {n_gene_edges} gene annotations")

    # Validate with networkx
    try:
        import networkx as nx
        G = nx.DiGraph((p, c) for p, c, r in ontology_rows if r == "default")
        roots = [n for n in G.nodes if G.in_degree(n) == 0]
        n_comp = nx.number_connected_components(G.to_undirected())
        if len(roots) == 1 and n_comp == 1:
            print(f"  ✓ Single root ({roots[0]}), connected hierarchy")
        else:
            print(f"  ⚠ {len(roots)} root(s), {n_comp} connected component(s) — "
                  f"gene filtering may have fragmented the hierarchy. "
                  f"Training will validate and report errors.")
    except ImportError:
        pass

    return gene2ind, ontology_rows


def save_gene2ind(gene2ind: dict, path: Path):
    with open(path, "w") as f:
        for gene, idx in sorted(gene2ind.items(), key=lambda x: x[1]):
            f.write(f"{idx}\t{gene}\n")


def save_ontology(ontology_rows: list, path: Path):
    with open(path, "w") as f:
        for parent, child, relation in ontology_rows:
            f.write(f"{parent}\t{child}\t{relation}\n")


def interactive_ontology_selection(
    mut_df: pd.DataFrame, cnv_df: pd.DataFrame,
    fusions_df: pd.DataFrame, sample_ids: list,
    default_uuid: str,
    ndex_uuid_arg: str | None,
    min_freq_arg: float | None,
    gene_list_arg: str | None,
    min_genes_arg: int | None = None,
) -> tuple[dict, list, dict] | None:
    """
    Ask the user how to select the gene panel and ontology hierarchy.
    Returns (gene2ind, ontology_rows, selection_metadata) for custom, or None to use default files.
    selection_metadata keys: ndex_uuid, min_alt_freq (float|None), gene_list_file (str|None),
                             gene_count, min_genes (int)
    """
    print("\n" + "=" * 60)
    print("GENE PANEL & HIERARCHY SELECTION")
    print("=" * 60)

    if ndex_uuid_arg:
        uuid = ndex_uuid_arg
    else:
        print("  [0] Use default  (data/gene2ind.txt + data/ontology.txt)")
        print("  [1] Build from NDEx hierarchy (custom gene set)")
        raw = input("\nChoice [0]: ").strip()
        if raw != "1":
            return None
        print(f"\nEnter NDEx hierarchy UUID (press Enter for default NeST hierarchy):")
        raw = input(f"  UUID [{default_uuid}]: ").strip()
        uuid = raw if raw else default_uuid

    # ── Gene set ──────────────────────────────────────────────────────────────
    gene_set: set[str] | None = None
    min_freq_used: float | None = None
    gene_list_file: str | None = None

    if gene_list_arg:
        with open(gene_list_arg) as fh:
            gene_set = {line.strip() for line in fh if line.strip()}
        gene_list_file = gene_list_arg
        print(f"\nLoaded {len(gene_set)} genes from {gene_list_arg}")

    elif min_freq_arg is not None:
        gene_set = compute_altered_genes(mut_df, cnv_df, fusions_df, sample_ids, min_freq_arg)
        min_freq_used = min_freq_arg
        print(f"\nGenes altered in >= {min_freq_arg*100:.1f}% of {len(sample_ids)} samples: "
              f"{len(gene_set)}")

    else:
        print("\nHow should the gene set be determined?")
        print("  [0] Frequency-based — genes altered in >= X% of this cohort (recommended)")
        print("  [1] From file       — provide a gene list (one symbol per line)")
        raw = input("\nChoice [0]: ").strip()

        if raw == "1":
            path_str = input("  Path to gene list file: ").strip()
            try:
                with open(path_str) as fh:
                    gene_set = {line.strip() for line in fh if line.strip()}
                gene_list_file = path_str
                print(f"  Loaded {len(gene_set)} genes from {path_str}")
            except FileNotFoundError:
                print(f"  ⚠ File not found — falling back to frequency-based selection.")

        if gene_set is None:
            print(f"\n  Computing alteration frequencies across {len(sample_ids)} samples ...")
            print(f"  {'Threshold':>10s}  {'Genes':>6s}")
            print(f"  {'─'*19}")
            for pct in [0.5, 1.0, 3.0, 5.0]:
                n = len(compute_altered_genes(mut_df, cnv_df, fusions_df, sample_ids, pct / 100))
                print(f"  {pct:>9.1f}%  {n:>6d}")

            raw_freq = input(
                f"\n  Minimum alteration frequency, e.g. '1%' or '0.01' "
                f"[default: 1%]: "
            ).strip()
            try:
                if "%" in raw_freq:
                    min_freq = float(raw_freq.rstrip("%")) / 100
                else:
                    min_freq = float(raw_freq) if raw_freq else 0.01
                    if min_freq > 1:          # user entered e.g. "5" meaning 5%
                        min_freq /= 100
            except ValueError:
                min_freq = 0.01
            min_freq_used = min_freq
            gene_set = compute_altered_genes(mut_df, cnv_df, fusions_df, sample_ids, min_freq)
            print(f"  → {len(gene_set)} genes at >= {min_freq*100:.1f}%")

    if min_genes_arg is not None:
        min_genes = min_genes_arg
    else:
        raw_mg = input(
            f"\n  Minimum panel genes per assembly [default: 5, per NeST-VNN paper]: "
        ).strip()
        try:
            min_genes = int(raw_mg) if raw_mg else 5
        except ValueError:
            min_genes = 5
    print(f"  → minimum genes per assembly: {min_genes}")

    gene2ind, ontology_rows = build_ontology_from_ndex(uuid, gene_set, min_genes=min_genes)
    selection_metadata = {
        "ndex_uuid":     uuid,
        "min_alt_freq":  min_freq_used,
        "gene_list_file": gene_list_file,
        "gene_count":    len(gene2ind),
        "min_genes":     min_genes,
    }
    return gene2ind, ontology_rows, selection_metadata


# ── Gene panel ───────────────────────────────────────────────────────────────

def load_gene_panel(gene2ind_path: Path) -> dict[str, int]:
    gene2ind = {}
    with open(gene2ind_path) as f:
        for line in f:
            idx, gene = line.strip().split("\t")
            gene2ind[gene] = int(idx)
    print(f"Loaded gene panel: {len(gene2ind)} genes")
    return gene2ind


# ── Helpers ──────────────────────────────────────────────────────────────────

def find_gene_col(df: pd.DataFrame) -> str | None:
    for candidate in ["gene.hugoGeneSymbol", "hugoGeneSymbol"]:
        if candidate in df.columns:
            return candidate
    return None


def save_matrix(matrix: np.ndarray, path: Path):
    with open(path, "w") as f:
        for row in matrix:
            f.write(",".join(str(x) for x in row) + "\n")


def save_tsv(df: pd.DataFrame, path: Path, header: bool = False):
    df.to_csv(path, sep="\t", index=False, header=header)


def build_cell2ind(sample_ids: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"idx": range(len(sample_ids)), "sample_id": sample_ids})


# ── Matrix builders ─────────────────────────────────────────────────────────

def build_mutation_matrix(mutations_df, sample_ids, gene2ind):
    n_samples, n_genes = len(sample_ids), len(gene2ind)
    matrix = np.zeros((n_samples, n_genes), dtype=int)
    sample2idx = {s: i for i, s in enumerate(sample_ids)}
    gene_col = find_gene_col(mutations_df)
    if gene_col is None:
        print("⚠ Cannot find gene symbol column in mutations.")
        return matrix
    mapped = 0
    for _, row in mutations_df.iterrows():
        s, g = row.get("sampleId"), row.get(gene_col)
        if s in sample2idx and g in gene2ind:
            matrix[sample2idx[s], gene2ind[g]] = 1
            mapped += 1
    panel_hit = int(np.sum(matrix.any(axis=0)))
    print(f"  Mutations:        {mapped}/{len(mutations_df)} mapped ({panel_hit}/{n_genes} genes hit)")
    return matrix


def build_cnv_matrices(cnv_df, sample_ids, gene2ind):
    n_samples, n_genes = len(sample_ids), len(gene2ind)
    del_m = np.zeros((n_samples, n_genes), dtype=int)
    amp_m = np.zeros((n_samples, n_genes), dtype=int)
    if cnv_df.empty:
        print("  CNV: no data")
        return del_m, amp_m
    sample2idx = {s: i for i, s in enumerate(sample_ids)}
    gene_col = find_gene_col(cnv_df)
    if gene_col is None:
        print("⚠ Cannot find gene symbol column in CNV.")
        return del_m, amp_m
    val_col = "alteration" if "alteration" in cnv_df.columns else "value"
    if val_col not in cnv_df.columns:
        print("⚠ Cannot find alteration value column in CNV.")
        return del_m, amp_m
    dc, ac = 0, 0
    for _, row in cnv_df.iterrows():
        s, g, v = row.get("sampleId"), row.get(gene_col), row.get(val_col)
        if s not in sample2idx or g not in gene2ind:
            continue
        try:
            v = int(float(v))
        except (ValueError, TypeError):
            continue
        si, gi = sample2idx[s], gene2ind[g]
        if v <= CNV_DEEP_DELETION:
            del_m[si, gi] = 1; dc += 1
        if v >= CNV_AMPLIFICATION:
            amp_m[si, gi] = 1; ac += 1
    del_hit = int(np.sum(del_m.any(axis=0)))
    amp_hit = int(np.sum(amp_m.any(axis=0)))
    print(f"  CN deletions:     {dc} events ({del_hit}/{n_genes} genes hit)")
    print(f"  CN amplifications: {ac} events ({amp_hit}/{n_genes} genes hit)")
    return del_m, amp_m


def build_fusion_matrix(fusions_df, sample_ids, gene2ind):
    n_samples, n_genes = len(sample_ids), len(gene2ind)
    matrix = np.zeros((n_samples, n_genes), dtype=int)
    if fusions_df.empty:
        print("  Fusions: no data")
        return matrix
    sample2idx = {s: i for i, s in enumerate(sample_ids)}
    sv_cols = [c for c in ["site1HugoSymbol", "site2HugoSymbol",
                           "site1.hugoSymbol", "site2.hugoSymbol",
                           "gene1.hugoGeneSymbol", "gene2.hugoGeneSymbol"]
               if c in fusions_df.columns]
    has_sv = len(sv_cols) > 0
    ev = 0
    for _, row in fusions_df.iterrows():
        s = row.get("sampleId")
        if s not in sample2idx:
            continue
        si = sample2idx[s]
        genes = []
        if has_sv:
            for c in sv_cols:
                g = row.get(c)
                if isinstance(g, str) and g.strip():
                    genes.append(g.strip())
        else:
            gc = find_gene_col(fusions_df)
            if gc:
                g = row.get(gc)
                if isinstance(g, str) and g.strip():
                    genes.append(g.strip())
        hit = False
        for g in genes:
            if g in gene2ind:
                matrix[si, gene2ind[g]] = 1; hit = True
        if hit:
            ev += 1
    panel_hit = int(np.sum(matrix.any(axis=0)))
    print(f"  Fusions:          {ev}/{len(fusions_df)} events ({panel_hit}/{n_genes} genes hit)")
    return matrix


# ── Training data builder ────────────────────────────────────────────────────


def build_training_data(clinical_df, sample_ids, endpoints, covariates=None):
    if clinical_df.empty or not endpoints:
        return pd.DataFrame()

    # Build a lookup: try sampleId first, then patientId
    # The clinical data may have both columns, or just patientId
    clin_by_sample = {}
    clin_by_patient = {}

    if "sampleId" in clinical_df.columns:
        for _, row in clinical_df.iterrows():
            clin_by_sample[row["sampleId"]] = row
    if "patientId" in clinical_df.columns:
        for _, row in clinical_df.iterrows():
            pid = row["patientId"]
            if pid not in clin_by_patient:  # keep first occurrence
                clin_by_patient[pid] = row

    # Also build sampleId → patientId from genomic data if clinical has patientId
    # Many studies have sampleId = patientId + suffix, but the pattern varies.
    # We try: exact sampleId match, then check if sampleId starts with any patientId.
    sample_to_patient = {}
    if clin_by_patient:
        patient_ids = sorted(clin_by_patient.keys(), key=len, reverse=True)  # longest first
        for sid in sample_ids:
            if sid in clin_by_sample:
                continue  # direct match available
            for pid in patient_ids:
                if sid.startswith(pid):
                    sample_to_patient[sid] = pid
                    break

    def resolve(sid):
        if sid in clin_by_sample:
            return clin_by_sample[sid]
        if sid in sample_to_patient:
            return clin_by_patient[sample_to_patient[sid]]
        if sid in clin_by_patient:
            return clin_by_patient[sid]
        return None

    # Debug: report match rate
    matched = sum(1 for sid in sample_ids if resolve(sid) is not None)
    print(f"\n  Clinical match: {matched}/{len(sample_ids)} samples linked to clinical records")
    if matched == 0:
        # Show what IDs look like to help debug
        sample_examples = sample_ids[:3]
        patient_examples = list(clin_by_patient.keys())[:3] if clin_by_patient else []
        sample_ex = list(clin_by_sample.keys())[:3] if clin_by_sample else []
        print(f"    Sample IDs (mutations):  {sample_examples}")
        print(f"    Patient IDs (clinical):  {patient_examples}")
        print(f"    Sample IDs (clinical):   {sample_ex}")

    rows = []
    for sid in sample_ids:
        record = resolve(sid)
        if record is None:
            continue
        row = {"cell_line": sid}
        has_any = False

        for ep in endpoints:
            raw = record.get(ep["clinical_col"], None)
            if raw is None or (isinstance(raw, float) and pd.isna(raw)):
                row[ep["label_name"]] = None
                continue
            raw_str = str(raw).strip()
            if raw_str == "":
                row[ep["label_name"]] = None
                continue

            if ep["mode"] == "continuous":
                try:
                    row[ep["label_name"]] = float(raw_str)
                    has_any = True
                except (ValueError, TypeError):
                    row[ep["label_name"]] = None

            elif ep["mode"] == "binary_status":
                mapping = ep["mapping"]
                val = mapping.get(raw_str)
                if val is None:
                    # Try case-insensitive match
                    for k, v in mapping.items():
                        if k.upper() == raw_str.upper():
                            val = v
                            break
                row[ep["label_name"]] = val
                if val is not None:
                    has_any = True

            elif ep["mode"] == "binary_threshold":
                try:
                    num = float(raw_str)
                    row[ep["label_name"]] = 1.0 if num > ep["mapping"]["threshold"] else 0.0
                    has_any = True
                except (ValueError, TypeError):
                    row[ep["label_name"]] = None

        row["dataset"] = "cBioPortal"
        if has_any:
            rows.append(row)

    col_order = ["cell_line"] + [ep["label_name"] for ep in endpoints] + ["dataset"]
    df = pd.DataFrame(rows, columns=col_order)

    if covariates:
        cov_arrays = build_covariates(clinical_df, sample_ids, covariates)
        cov_df = pd.DataFrame({"cell_line": sample_ids})
        for label_name, arr in cov_arrays.items():
            cov_df[label_name] = arr
        df = df.merge(cov_df, on="cell_line", how="left")
        cov_col_names = [cov["label_name"] for cov in covariates]
        df = df[["cell_line"] + [ep["label_name"] for ep in endpoints] + cov_col_names + ["dataset"]]

    print(f"\n  Training data: {len(df)} samples")
    for ep in endpoints:
        valid = df[ep["label_name"]].dropna()
        if valid.empty:
            print(f"    {ep['label_name']:30s}  → no data")
        elif ep["mode"] == "continuous":
            print(f"    {ep['label_name']:30s}  → {len(valid)} values  "
                  f"range=[{valid.min():.1f}, {valid.max():.1f}]  median={valid.median():.1f}")
        else:
            n0 = int((valid == 0).sum())
            n1 = int((valid == 1).sum())
            print(f"    {ep['label_name']:30s}  → {len(valid)} values  event(0)={n0}  no-event(1)={n1}")

    return df


# ── README ───────────────────────────────────────────────────────────────────

def write_readme(output_dir, sample_ids, gene2ind, endpoints,
                 mut_matrix, del_matrix, amp_matrix, fus_matrix, training_df,
                 metadata: dict | None = None):
    lines = []
    w = lines.append
    n_s, n_g = len(sample_ids), len(gene2ind)

    w("# NeST-VNN Input Data")
    w("")
    w(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    w("")
    w(f"- **Samples:** {n_s}")
    w(f"- **Gene panel:** {n_g} genes")
    w("")

    if metadata:
        w("## Data Sources")
        w("")
        if "cbioportal_url" in metadata:
            w(f"- **cBioPortal study:** [{metadata['study_id']}]({metadata['cbioportal_url']})")
        if "ndex_uuid" in metadata:
            w(f"- **Ontology (NDEx):** [{metadata['ndex_uuid']}]({metadata['ndex_url']})")
            w(f"- **Min genes per assembly:** {metadata.get('min_genes', 5)} "
              f"(paper default: 5)")
            if metadata.get("min_alt_freq") is not None:
                w(f"- **Gene selection:** frequency-filtered — genes altered in "
                  f"≥ {metadata['min_alt_freq']*100:.1f}% of {n_s} samples "
                  f"({n_g} genes retained)")
            elif metadata.get("gene_list_file"):
                w(f"- **Gene selection:** from file `{metadata['gene_list_file']}` "
                  f"({n_g} genes matched in ontology)")
        else:
            w(f"- **Ontology:** default (`data/ontology.txt`)")
            w(f"- **Gene panel:** default (`data/gene2ind.txt`, {n_g} genes)")
        w("")

    any_alt = (mut_matrix | del_matrix | amp_matrix | fus_matrix)
    w("## Feature Matrices")
    w("")
    w("| Feature | Events | Genes hit | Density |")
    w("|---|---|---|---|")
    for name, mat in [("Mutations", mut_matrix), ("CN Deletions", del_matrix),
                      ("CN Amplifications", amp_matrix), ("Fusions", fus_matrix)]:
        genes_hit = int(np.sum(mat.any(axis=0)))
        w(f"| {name} | {int(mat.sum())} | {genes_hit}/{n_g} | {100*mat.mean():.2f}% |")
    w(f"| **Any alteration** | {int(any_alt.sum())} | "
      f"{int(np.sum(any_alt.any(axis=0)))}/{n_g} | "
      f"samples={int(np.sum(any_alt.any(axis=1)))}/{n_s}, {100*any_alt.mean():.2f}% |")
    w("")

    if not training_df.empty and endpoints:
        w("## Endpoints")
        w("")
        w("| Label | Mode | N | Summary |")
        w("|---|---|---|---|")
        for ep in endpoints:
            col = ep["label_name"]
            if col not in training_df.columns:
                continue
            valid = training_df[col].dropna()
            if valid.empty:
                w(f"| `{col}` | {ep['mode']} | 0 | no data |")
            elif ep["mode"] == "continuous":
                w(f"| `{col}` | continuous | {len(valid)} | "
                  f"median={valid.median():.1f}, [{valid.min():.1f}, {valid.max():.1f}] |")
            else:
                n0, n1 = int((valid==0).sum()), int((valid==1).sum())
                w(f"| `{col}` | binary | {len(valid)} | event(0)={n0}, no-event(1)={n1} |")
        w("")

        w("## Endpoint Configuration")
        w("")
        w("```json")
        w(json.dumps(endpoints, indent=2))
        w("```")
        w("")

    w("## Usage")
    w("")
    if endpoints:
        ep0 = endpoints[0]
        task = "binary" if "binary" in ep0["mode"] else "continuous"
        w("```bash")
        w(f"python src/train.py \\")
        w(f"  -train {output_dir}/training_data.txt \\")
        w(f"  -label {ep0['label_name']} -task {task} \\")
        w(f"  -mutations {output_dir}/cell2mutation.txt \\")
        w(f"  -cn_deletions {output_dir}/cell2cndeletion.txt \\")
        w(f"  -cn_amplifications {output_dir}/cell2cnamplification.txt \\")
        w(f"  -fusions {output_dir}/cell2fusion.txt \\")
        w(f"  -onto {output_dir}/ontology.txt \\")
        w(f"  -gene2id {output_dir}/gene2ind.txt \\")
        w(f"  -cell2id {output_dir}/cell2ind.txt \\")
        w(f"  -std MODEL/std.txt -model MODEL/ \\")
        w(f"  -cuda 0 -epoch 300 -batchsize 64")
        w("```")
    w("")

    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"  ✓ README.md")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Transform cBioPortal data → NeST-VNN format")
    parser.add_argument("study_id", help="cBioPortal study ID (e.g. laml_tcga_pub, breast_msk_2025)")
    parser.add_argument("--data-dir", help="Base data directory", default="data")
    parser.add_argument("--endpoints-json", help="JSON file with endpoint config (skip interactive)", default=None)
    parser.add_argument("--ndex-uuid", help=f"NDEx hierarchy UUID (skips interactive prompt; default NeST: {DEFAULT_NDEX_UUID})", default=None)
    parser.add_argument("--min-alt-freq", type=float, help="Min alteration frequency 0–1 for gene selection with --ndex-uuid (default 0.01)", default=None)
    parser.add_argument("--gene-list", help="Path to gene list file (one symbol per line) for use with --ndex-uuid", default=None)
    parser.add_argument("--min-genes", type=int, default=None,
                        help="Minimum panel genes per assembly when building from NDEx (default: 5, per NeST-VNN paper)")
    args = parser.parse_args()

    study_id = args.study_id
    data_dir = Path(args.data_dir)
    study_dir = data_dir / "output" / study_id

    input_dir = study_dir / "cbioportal_output"
    output_dir = study_dir / "nest_vnn_input"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Fallback gene panel / ontology (used when user chooses default)
    gene2ind_path = data_dir / "gene2ind.txt"
    ontology_path = data_dir / "ontology.txt"

    print("=" * 60)
    print("cBioPortal → NeST-VNN Format Converter")
    print("=" * 60)
    print(f"Study:    {study_id}")
    print(f"Input:    {input_dir}")
    print(f"Output:   {output_dir}\n")

    mutations_path = input_dir / "mutations.csv"
    if not mutations_path.exists():
        print(f"ERROR: {mutations_path} not found. Run cbioportal_download.py {study_id} first.")
        sys.exit(1)

    # Load genomic + clinical data
    print("Loading data ...")
    mutations_df = pd.read_csv(mutations_path, low_memory=False)
    print(f"  mutations:  {len(mutations_df)} rows")

    cnv_path = input_dir / "cnv.csv"
    cnv_df = pd.read_csv(cnv_path, low_memory=False) if cnv_path.exists() else pd.DataFrame()
    print(f"  cnv:        {len(cnv_df)} rows")

    fusions_path = input_dir / "fusions.csv"
    fusions_df = pd.read_csv(fusions_path, low_memory=False) if fusions_path.exists() else pd.DataFrame()
    print(f"  fusions:    {len(fusions_df)} rows")

    clinical_path = input_dir / "clinical_outcomes.csv"
    if not clinical_path.exists():
        clinical_path = input_dir / "patient_clinical.csv"
    clinical_df = pd.read_csv(clinical_path, low_memory=False) if clinical_path.exists() else pd.DataFrame()
    print(f"  clinical:   {len(clinical_df)} rows ({clinical_path.name})")

    sample_ids = sorted(mutations_df["sampleId"].unique())
    print(f"\nTotal unique samples: {len(sample_ids)}")

    # ── Gene panel & ontology selection ──────────────────────────────────────
    ndex_result = interactive_ontology_selection(
        mutations_df, cnv_df, fusions_df, sample_ids,
        default_uuid=DEFAULT_NDEX_UUID,
        ndex_uuid_arg=args.ndex_uuid,
        min_freq_arg=args.min_alt_freq,
        gene_list_arg=args.gene_list,
        min_genes_arg=args.min_genes,
    )

    ndex_metadata: dict | None = None
    if ndex_result is not None:
        gene2ind, ontology_rows, ndex_metadata = ndex_result
        use_default_ontology = False
    else:
        for p in [gene2ind_path, ontology_path]:
            if not p.exists():
                print(f"ERROR: {p} not found.")
                print(f"Copy gene2ind.txt and ontology.txt to {data_dir}/ or use --ndex-uuid.")
                sys.exit(1)
        gene2ind = load_gene_panel(gene2ind_path)
        ontology_rows = None
        use_default_ontology = True
        print(f"\nGene panel: {len(gene2ind)} genes from {gene2ind_path}")

    # ── Endpoint selection ────────────────────────────────────────────────────
    if args.endpoints_json:
        with open(args.endpoints_json) as f:
            endpoints = json.load(f)
        print(f"\nLoaded {len(endpoints)} endpoints from {args.endpoints_json}")
    else:
        endpoints = interactive_endpoint_selection(clinical_df)

    if endpoints:
        config_path = output_dir / "endpoints.json"
        with open(config_path, "w") as f:
            json.dump(endpoints, f, indent=2)
        print(f"\nEndpoint config saved to {config_path} (reuse with --endpoints-json)")

    # ── Covariate selection ───────────────────────────────────────────────────
    endpoint_clinical_cols = {ep["clinical_col"] for ep in endpoints}
    covariates = interactive_covariate_selection(clinical_df, endpoint_clinical_cols)

    # ── Build matrices ────────────────────────────────────────────────────────
    print("\nBuilding NeST-VNN matrices ...")
    cell2ind_df = build_cell2ind(sample_ids)
    mut_matrix = build_mutation_matrix(mutations_df, sample_ids, gene2ind)
    del_matrix, amp_matrix = build_cnv_matrices(cnv_df, sample_ids, gene2ind)
    fus_matrix = build_fusion_matrix(fusions_df, sample_ids, gene2ind)
    training_df = build_training_data(clinical_df, sample_ids, endpoints, covariates=covariates)

    # ── Write output files ────────────────────────────────────────────────────
    print(f"\nWriting to {output_dir}/ ...")

    save_tsv(cell2ind_df, output_dir / "cell2ind.txt")
    print(f"  ✓ cell2ind.txt          ({len(cell2ind_df)} samples)")

    if use_default_ontology:
        shutil.copy(gene2ind_path, output_dir / "gene2ind.txt")
        print(f"  ✓ gene2ind.txt          (from {gene2ind_path})")
    else:
        save_gene2ind(gene2ind, output_dir / "gene2ind.txt")
        print(f"  ✓ gene2ind.txt          ({len(gene2ind)} genes from NDEx)")

    save_matrix(mut_matrix, output_dir / "cell2mutation.txt")
    print(f"  ✓ cell2mutation.txt     ({mut_matrix.shape})")

    save_matrix(del_matrix, output_dir / "cell2cndeletion.txt")
    print(f"  ✓ cell2cndeletion.txt   ({del_matrix.shape})")

    save_matrix(amp_matrix, output_dir / "cell2cnamplification.txt")
    print(f"  ✓ cell2cnamplification.txt ({amp_matrix.shape})")

    save_matrix(fus_matrix, output_dir / "cell2fusion.txt")
    print(f"  ✓ cell2fusion.txt       ({fus_matrix.shape})")

    if use_default_ontology:
        shutil.copy(ontology_path, output_dir / "ontology.txt")
        print(f"  ✓ ontology.txt          (from {ontology_path})")
    else:
        save_ontology(ontology_rows, output_dir / "ontology.txt")
        print(f"  ✓ ontology.txt          ({len(ontology_rows)} rows from NDEx)")

    if not training_df.empty:
        training_df.to_csv(output_dir / "training_data.txt", sep="\t", index=False)
        print(f"  ✓ training_data.txt     ({len(training_df)} rows)")

    if covariates:
        with open(output_dir / "covariate_stats.json", "w") as f:
            json.dump(covariates, f, indent=2)
        print(f"  ✓ covariate_stats.json  ({len(covariates)} covariates)")

    std_df = pd.DataFrame({"source": ["cBioPortal"], "mean": [0.0], "std": [1.0]})
    save_tsv(std_df, output_dir / "std.txt")
    print(f"  ✓ std.txt")

    # Write metadata.json for downstream MLflow logging
    readme_metadata: dict = {
        "study_id":       study_id,
        "cbioportal_url": f"https://www.cbioportal.org/study/summary?id={study_id}",
    }
    if ndex_metadata:
        uuid = ndex_metadata["ndex_uuid"]
        readme_metadata.update({
            "ndex_uuid":    uuid,
            "ndex_url":     f"https://www.ndexbio.org/viewer/networks/{uuid}",
            "min_alt_freq": ndex_metadata.get("min_alt_freq"),
            "gene_list_file": ndex_metadata.get("gene_list_file"),
            "gene_count":   ndex_metadata["gene_count"],
        })
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(readme_metadata, f, indent=2)
    print(f"  ✓ metadata.json")

    write_readme(output_dir, sample_ids, gene2ind, endpoints,
                 mut_matrix, del_matrix, amp_matrix, fus_matrix, training_df,
                 metadata=readme_metadata)

    print(f"\n{'='*60}")
    print(f"DONE. Output in: {output_dir.resolve()}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()