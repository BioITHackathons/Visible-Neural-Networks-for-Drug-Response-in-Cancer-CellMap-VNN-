"""
CFDE IDG (Illuminating the Druggable Genome) gene→drug lookup via Pharos/TCRD.

Source: Pharos GraphQL API (https://pharos-api.ncats.io/graphql)
Dataset: CFDE IDG — https://cfdeknowledge.org/r/kc_programs?DCC=IDG

Only Tclin targets are included. Tclin = genes with FDA-approved drugs recorded in
DrugCentral, which is the authoritative CFDE IDG drug-target resource. Genes classified
as Tchem, Tbio, or Tdark have no CFDE-approved drugs and are omitted from the lookup.

Usage:
    python interpretation/download_cfde.py          # download + save JSON
    from interpretation.cfde_drugs import get_drugs  # lookup at runtime
"""

import glob
import json
import os
import re
import time
import requests
from pathlib import Path

PHAROS_URL = "https://pharos-api.ncats.io/graphql"
CACHE_PATH  = Path(__file__).parent / "cfde_drug_lookup.json"
_DATA_ROOT  = Path(__file__).parent.parent / "data" / "output"

_QUERY = """
query getTarget($sym: String!) {
  target(q: {sym: $sym}) {
    sym
    tdl
    ligands(isdrug: true, top: 20) {
      ligid
      name
    }
  }
}
"""

# Salt forms and hydrates to strip from drug names
_SALT_RE = re.compile(
    r"\s+(hydrochloride|hcl|anhydrous|monohydrate|dihydrate|mesylate|"
    r"tosylate|maleate|dimaleate|fumarate|succinate|acetate|phosphate|"
    r"sulfate|nitrate|citrate|tartrate|besylate|pamoate|sodium|potassium)$",
    re.IGNORECASE,
)


def _clean_name(raw: str) -> str:
    return _SALT_RE.sub("", raw.strip().title()).strip()


# FDA-approved drugs with breast cancer indications (DrugCentral / NCCN)
_BREAST_CANCER_DRUGS: frozenset[str] = frozenset({
    # CDK4/6 inhibitors
    "abemaciclib", "palbociclib", "ribociclib",
    # HER2-targeted
    "trastuzumab", "pertuzumab", "trastuzumab emtansine", "trastuzumab deruxtecan",
    "lapatinib", "neratinib", "tucatinib", "margetuximab",
    # Endocrine therapy (SERMs, SERDs, aromatase inhibitors)
    "tamoxifen", "toremifene", "raloxifene", "fulvestrant", "elacestrant",
    "letrozole", "anastrozole", "exemestane",
    # PARP inhibitors (BRCA-mutated breast cancer)
    "olaparib", "talazoparib",
    # PI3K / AKT / mTOR inhibitors
    "alpelisib", "inavolisib", "capivasertib", "everolimus",
    # Immunotherapy (triple-negative breast cancer)
    "pembrolizumab", "atezolizumab",
    # Antibody-drug conjugates
    "sacituzumab govitecan", "datopotamab deruxtecan",
    # Other breast-cancer approved agents
    "bevacizumab", "eribulin", "capecitabine", "ixabepilone",
})


def filter_breast_cancer(drugs: list[str]) -> list[str]:
    """Return only drugs with documented breast cancer indications."""
    return [d for d in drugs if d.lower() in _BREAST_CANCER_DRUGS]


def _find_gene_list() -> list[str]:
    """Auto-discover gene symbols from the most recently created gene2ind.txt."""
    pattern = str(_DATA_ROOT / "**" / "nest_vnn_input" / "gene2ind.txt")
    files = sorted(glob.glob(pattern, recursive=True), key=os.path.getmtime, reverse=True)
    if not files:
        return []
    genes = []
    with open(files[0]) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                genes.append(parts[1].strip())
    return genes


def download(
    genes: list[str] | None = None,
    out_path: Path = CACHE_PATH,
) -> dict:
    """
    Query Pharos (CFDE IDG) for each gene and save gene→[drug_names] as JSON.

    Only Tclin genes (with FDA-approved drugs in DrugCentral) are included.
    Tchem/Tbio/Tdark genes are skipped — that is the correct CFDE classification.

    genes: list of gene symbols. If None, auto-discovers from gene2ind.txt.
    Returns {GENE_SYMBOL: [drug_name, ...]}
    """
    if genes is None:
        genes = _find_gene_list()
    if not genes:
        raise RuntimeError(
            "No genes found. Either pass a list or run the training pipeline first "
            "so gene2ind.txt exists under data/output/."
        )

    print(f"[cfde-idg] Querying Pharos for {len(genes)} genes ...")

    lookup: dict[str, list[str]] = {}
    tdl_counts: dict[str, int] = {}
    session = requests.Session()
    session.headers.update({"Content-Type": "application/json"})

    for i, gene in enumerate(genes):
        try:
            r = session.post(
                PHAROS_URL,
                json={"query": _QUERY, "variables": {"sym": gene}},
                timeout=30,
            )
            r.raise_for_status()
            payload = r.json()
        except requests.RequestException as e:
            print(f"[cfde-idg]   WARNING: request failed for {gene}: {e}")
            time.sleep(1.0)
            continue

        if "errors" in payload:
            continue

        t = (payload.get("data") or {}).get("target")
        if not t:
            continue

        tdl = t.get("tdl") or "unknown"
        tdl_counts[tdl] = tdl_counts.get(tdl, 0) + 1

        if tdl == "Tclin":
            drugs = [
                _clean_name(lig["name"])
                for lig in (t.get("ligands") or [])
                if lig.get("name") and lig["name"].strip()
            ]
            drugs = [d for d in drugs if d]  # drop empty after cleaning
            if drugs:
                lookup[gene.upper()] = drugs

        if (i + 1) % 50 == 0 or (i + 1) == len(genes):
            print(f"[cfde-idg]   {i+1}/{len(genes)} — {len(lookup)} Tclin genes with drugs so far")

        time.sleep(0.2)

    out_path.write_text(json.dumps(lookup, indent=2))
    print(f"[cfde-idg] Saved {out_path}")
    return lookup, tdl_counts


def load(path: Path = CACHE_PATH) -> dict:
    """Load cached lookup. Returns empty dict if not yet downloaded."""
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def get_drugs(genes: list[str], lookup: dict | None = None) -> dict[str, list[str]]:
    """
    Return {gene: [drug_names]} for Tclin genes in the CFDE IDG lookup.
    Genes not in the Tclin catalogue are omitted (correct CFDE behaviour).
    """
    if lookup is None:
        lookup = load()
    return {g: lookup[g.upper()] for g in genes if g.upper() in lookup}
