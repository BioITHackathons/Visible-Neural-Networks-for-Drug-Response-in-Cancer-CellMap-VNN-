"""
CFDE Druggable Genome gene→drug lookup via DGIdb (Drug Gene Interaction Database).

DGIdb is the CFDE-linked resource for clinical drug-gene interactions in cancer.
Unlike Pharos Tclin (direct binding only), DGIdb captures:
  - Targeted therapies approved for patients with a given gene alteration
  - Synthetic lethal relationships (e.g., PARP inhibitors for BRCA1/2 mutations)
  - Pathway-mediated targeting

Only FDA-approved drugs (drug.approved = true) are included.

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

DGIDB_URL  = "https://dgidb.org/api/graphql"
CACHE_PATH = Path(__file__).parent / "cfde_drug_lookup.json"

_DATA_ROOT = Path(__file__).parent.parent / "data" / "output"

_QUERY = """
query getGenes($names: [String!]!) {
  genes(names: $names) {
    nodes {
      name
      interactions {
        drug {
          name
          approved
        }
        interactionScore
        interactionTypes {
          type
        }
        sources {
          sourceDbName
        }
      }
    }
  }
}
"""

# Sources that are oncology-specific knowledgebases.
# An interaction must come from at least one of these to be included.
_ONCOLOGY_SOURCES = {
    "OncoKB", "CIViC", "MyCancerGenome", "MyCancerGenomeClinicalTrial",
    "CGI", "FDA", "DoCM", "COSMIC", "ClearityFoundationBiomarkers",
    "TALC", "CancerCommons", "TdgClinicalTrial",
}

# Salt forms, hydrates, and other noise suffixes to strip
_SALT_RE = re.compile(
    r"\s+(hydrochloride|hcl|anhydrous|monohydrate|dihydrate|mesylate|"
    r"tosylate|maleate|dimaleate|fumarate|succinate|acetate|phosphate|"
    r"sulfate|nitrate|citrate|tartrate|besylate|pamoate|sodium|potassium)$",
    re.IGNORECASE,
)


def _clean_name(raw: str) -> str:
    """Normalise drug name: title-case, strip salt forms."""
    name = raw.strip().title()
    name = _SALT_RE.sub("", name).strip()
    return name


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
    batch_size: int = 50,
) -> dict:
    """
    Query DGIdb for approved drug interactions for each gene and save as JSON.

    genes: list of gene symbols to query. If None, auto-discovers from gene2ind.txt.
    Returns {GENE_SYMBOL: [drug_name, ...]} for genes with approved drugs.
    """
    if genes is None:
        genes = _find_gene_list()
    if not genes:
        raise RuntimeError(
            "No genes found. Either pass a list or run the training pipeline first "
            "so gene2ind.txt exists under data/output/."
        )

    genes = [g.upper() for g in genes]
    print(f"[cfde] querying DGIdb for {len(genes)} genes in batches of {batch_size} ...")

    lookup: dict[str, list[str]] = {}
    session = requests.Session()
    session.headers.update({"Content-Type": "application/json"})

    for i in range(0, len(genes), batch_size):
        batch = genes[i : i + batch_size]
        try:
            r = session.post(
                DGIDB_URL,
                json={"query": _QUERY, "variables": {"names": batch}},
                timeout=60,
            )
            r.raise_for_status()
        except requests.RequestException as e:
            print(f"[cfde]   WARNING: request failed for batch {i//batch_size + 1}: {e}")
            continue

        payload = r.json()
        if "errors" in payload:
            print(f"[cfde]   WARNING: GraphQL errors for batch {i//batch_size + 1}: {payload['errors']}")
            continue

        for node in payload["data"]["genes"]["nodes"]:
            sym = node["name"].upper()
            # Collect (score, name) for approved drugs backed by an oncology source
            candidates: list[tuple[float, str]] = []
            for ix in node.get("interactions") or []:
                drug    = ix.get("drug") or {}
                score   = float(ix.get("interactionScore") or 0)
                sources = {s["sourceDbName"] for s in (ix.get("sources") or [])}

                if not drug.get("approved") or not drug.get("name"):
                    continue
                if not sources.intersection(_ONCOLOGY_SOURCES):
                    continue  # only accept interactions from oncology databases

                cleaned = _clean_name(drug["name"])
                if cleaned:
                    candidates.append((score, cleaned))

            if candidates:
                # Sort by score descending, deduplicate, keep top 15
                seen: set[str] = set()
                ranked: list[str] = []
                for _, name in sorted(candidates, key=lambda x: -x[0]):
                    if name not in seen:
                        seen.add(name)
                        ranked.append(name)
                    if len(ranked) == 15:
                        break
                if ranked:
                    lookup[sym] = ranked

        done = min(i + batch_size, len(genes))
        hits = sum(1 for g in batch if g in lookup)
        print(f"[cfde]   batch {i//batch_size + 1}: {done}/{len(genes)} genes — "
              f"{hits}/{len(batch)} in this batch have approved drugs")
        time.sleep(0.3)

    out_path.write_text(json.dumps(lookup, indent=2))
    print(f"[cfde] saved {len(lookup)} gene→drug mappings to {out_path}")
    return lookup


def load(path: Path = CACHE_PATH) -> dict:
    """Load cached lookup. Returns empty dict if not yet downloaded."""
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def get_drugs(genes: list[str], lookup: dict | None = None) -> dict[str, list[str]]:
    """
    Return {gene: [drug_names]} for genes present in the CFDE/DGIdb lookup.
    Genes with no approved drugs are omitted.
    """
    if lookup is None:
        lookup = load()
    return {g: lookup[g.upper()] for g in genes if g.upper() in lookup}
