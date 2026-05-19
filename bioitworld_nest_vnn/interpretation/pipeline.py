"""Orchestrate extract → batch → LLM → parse → store."""

import json
import re
from pathlib import Path

from . import db as _db
from . import llm_client as _llm
from . import cfde_drugs as _cfde
from .extractor import extract_patients
from .prompts import SYSTEM_PROMPT, build_user_message

_DRUG_LOOKUP = _cfde.load()


# ── LLM response parser ───────────────────────────────────────────────────────

# Patient header: "# Patient P-0000123-T01-IM3"
_PATIENT_HDR_RE  = re.compile(r"^#\s*Patient\s+(\S+)", re.MULTILINE)
# NEST section:   "## NEST 1: NEST:33"
_NEST_HDR_RE     = re.compile(r"^##\s*NEST\s+\d+:\s*(NEST:\d+)", re.MULTILINE | re.IGNORECASE)
# ### Summary block up to the next ### or ##
_SUMMARY_BLK_RE  = re.compile(r"###\s*Summary\s*\n(.*?)(?=###|^##\s)", re.DOTALL | re.IGNORECASE | re.MULTILINE)
# ### Top Genes block
_GENES_BLK_RE    = re.compile(r"###\s*Top Genes\s*\n(.*?)(?=^##\s|\Z)", re.DOTALL | re.IGNORECASE | re.MULTILINE)
# Each gene entry under ####
_GENE_ENTRY_RE   = re.compile(r"^####\s*([\w\-]+)\s*\n(.*?)(?=^####|\Z)", re.DOTALL | re.MULTILINE)
# Overall interpretation
_OVERALL_RE      = re.compile(r"^##\s*Overall Patient Interpretation\s*\n(.*?)(?=---|^#\s*Patient|\Z)", re.DOTALL | re.MULTILINE | re.IGNORECASE)
# Markdown link [name](url)
_MD_LINK_RE      = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
# Direction from summary prose
_DIRECTION_RE    = re.compile(r"associated with\s+(improved|worsened)", re.IGNORECASE)
# Alteration + direction from gene prose
_GENE_ALT_RE     = re.compile(r"showed\s+(.+?),\s*which\s+(improved|worsened)", re.IGNORECASE)
# Biological role from gene prose
_GENE_ROLE_RE    = re.compile(r"involved in\s+(.+?)(?:\.|$)", re.IGNORECASE)
# Drugs from gene prose
_GENE_DRUGS_RE   = re.compile(r"Potential therapies include\s+(.+?)(?:\(from|\.|$)", re.IGNORECASE)


def _clean(s: str) -> str:
    return s.strip().replace("\n", " ").replace("  ", " ") if s else ""


def _parse_gene_entry(gene_name: str, prose: str, orig_genes: dict) -> dict:
    """Parse a single #### GENE prose block into a structured dict."""
    alt_m   = _GENE_ALT_RE.search(prose)
    role_m  = _GENE_ROLE_RE.search(prose)
    drugs_m = _GENE_DRUGS_RE.search(prose)

    alteration = _clean(alt_m.group(1)) if alt_m else "not altered"
    direction  = alt_m.group(2).lower() if alt_m else ""
    bio_role   = _clean(role_m.group(1)) if role_m else ""
    drugs_raw  = _clean(drugs_m.group(1)) if drugs_m else ""
    drug_list  = [d.strip() for d in re.split(r"[,;]", drugs_raw)
                  if d.strip() and d.strip().lower() not in ("none known", "none", "")]

    og = orig_genes.get(gene_name, {})
    # Replace with CFDE-verified drugs if available
    cfde = _DRUG_LOOKUP.get(gene_name.upper())
    if cfde is not None:
        drug_list = cfde

    return {
        "gene_name":      gene_name,
        "alteration_type": alteration,
        "direction":       direction,
        "biological_role": bio_role,
        "drugs":           drug_list,
        "rank":            og.get("rank", 0),
        "importance_score": og.get("importance_score"),
    }


def _parse_llm_response(raw: str, patient_batch: list[dict]) -> list[dict]:
    """Parse LLM markdown response into structured per-patient dicts."""
    results = []
    patient_map = {p["patient_id"]: p for p in patient_batch}

    # Split on "# Patient <ID>" boundaries
    sections = re.split(r"(?=^#\s*Patient\s+)", raw, flags=re.MULTILINE)

    for section in sections:
        hdr = _PATIENT_HDR_RE.match(section)
        if not hdr:
            continue
        patient_id = hdr.group(1).strip()

        # Fuzzy fallback
        if patient_id not in patient_map:
            patient_id = next(
                (pid for pid in patient_map if pid in patient_id or patient_id in pid),
                patient_id,
            )

        base = patient_map.get(patient_id, patient_batch[0])
        orig_nests_map = {n["nest_id"]: n for n in base.get("top_nests", [])}

        # Overall interpretation
        overall_m = _OVERALL_RE.search(section)
        overall   = _clean(overall_m.group(1)) if overall_m else ""

        result = {
            "patient_id":             patient_id,
            "study_id":               base["study_id"],
            "label":                  base["label"],
            "predicted_probability":  base.get("predicted_probability"),
            "predicted_class":        base.get("predicted_class"),
            "sample_type":            base.get("sample_type"),
            "mutation_count":         base.get("mutation_count"),
            "fraction_genome_altered": base.get("fraction_genome_altered"),
            "llm_raw_markdown":       section.strip(),
            "clinical_summary":       overall,
            "nests":                  [],
        }

        # Find all NEST sections
        nest_positions = [(m.start(), m.group(1)) for m in _NEST_HDR_RE.finditer(section)]

        for idx, (pos, nest_id) in enumerate(nest_positions):
            end_pos = nest_positions[idx + 1][0] if idx + 1 < len(nest_positions) else len(section)
            # Stop before Overall Patient Interpretation
            overall_pos = (overall_m.start() if overall_m else len(section))
            end_pos = min(end_pos, overall_pos)
            nest_block = section[pos:end_pos]

            # --- Summary block ---
            summ_m     = _SUMMARY_BLK_RE.search(nest_block)
            summ_prose = summ_m.group(1) if summ_m else ""

            direction_m = _DIRECTION_RE.search(summ_prose)
            direction   = direction_m.group(1).lower() if direction_m else ""

            # Extract markdown link for pathway + reactome
            link_m    = _MD_LINK_RE.search(summ_prose)
            pathway   = _clean(link_m.group(1)) if link_m else ""
            reactome  = link_m.group(2).strip() if link_m else "N/A"
            if reactome and not reactome.startswith("http"):
                reactome = "N/A"

            # Split prose into bio explanation vs clinical reasoning on the sentence boundary
            bio_exp = clin = ""
            if summ_prose:
                sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", summ_prose.strip()) if s.strip()]
                # Sentence containing "may contribute" → clinical reasoning; rest → bio explanation
                bio_parts, clin_parts = [], []
                for s in sentences:
                    if re.search(r"may contribute|patient phenotype|because", s, re.IGNORECASE):
                        clin_parts.append(s)
                    else:
                        bio_parts.append(s)
                bio_exp = " ".join(bio_parts)
                clin    = " ".join(clin_parts)

            # --- Genes block ---
            genes_block = ""
            genes_m = _GENES_BLK_RE.search(nest_block)
            if genes_m:
                genes_block = genes_m.group(1)

            orig = orig_nests_map.get(nest_id, {})
            orig_genes = {g["gene_name"]: g for g in orig.get("top_patient_genes", [])}

            genes_parsed = []
            for gm in _GENE_ENTRY_RE.finditer(genes_block):
                gene_name = gm.group(1).strip()
                gene_prose = gm.group(2)
                genes_parsed.append(_parse_gene_entry(gene_name, gene_prose, orig_genes))

            result["nests"].append({
                "nest_id":               nest_id,
                "rank":                  orig.get("rank", idx + 1),
                "importance_score":      orig.get("importance_score"),
                "rlipp_score":           orig.get("rlipp_score"),
                "population_rlipp":      orig.get("population_rlipp"),
                "direction":             direction or orig.get("direction", ""),
                "pathway_name":          pathway,
                "reactome_link":         reactome,
                "biological_explanation": bio_exp,
                "clinical_reasoning":    clin,
                "genes":                 genes_parsed,
            })

        results.append(result)

    return results


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(
    study_id: str,
    label: str,
    batch_size: int = 5,
    limit: int = None,
    skip_existing: bool = True,
    db_path: Path = None,
):
    if db_path is None:
        from . import db as _db_mod
        db_path = _db_mod.DB_PATH

    _db.init_db(db_path)

    print(f"[pipeline] Extracting patients for study={study_id!r} label={label!r} ...")
    all_patients = extract_patients(study_id, label, limit=limit)
    print(f"[pipeline] {len(all_patients)} patients extracted.")

    if skip_existing:
        all_patients = [
            p for p in all_patients
            if not _db.already_processed(p["patient_id"], study_id, label, db_path)
        ]
        print(f"[pipeline] {len(all_patients)} patients need processing.")

    n_ok, n_err = 0, 0

    for batch_start in range(0, len(all_patients), batch_size):
        batch = all_patients[batch_start: batch_start + batch_size]
        ids = [p["patient_id"] for p in batch]
        print(f"[pipeline] Batch {batch_start//batch_size + 1}: {ids}")

        try:
            user_msg = build_user_message(batch)
            raw_response = _llm.call_llm(SYSTEM_PROMPT, user_msg)
            parsed = _parse_llm_response(raw_response, batch)
        except Exception as e:
            print(f"[pipeline] ERROR calling LLM for batch starting at {batch_start}: {e}")
            n_err += len(batch)
            continue

        with _db.get_conn(db_path) as conn:
            for rec in parsed:
                rec["llm_raw_markdown"] = rec.get("llm_raw_markdown", "")
                _db.upsert_patient(conn, rec)

                for nest in rec.get("nests", []):
                    _db.upsert_nest(conn, rec["patient_id"], study_id, label, nest)
                    for gene in nest.get("genes", []):
                        _db.upsert_gene(conn, rec["patient_id"], study_id, label, nest["nest_id"], gene)

            # Store any patients in the batch that weren't returned by the parser
            parsed_ids = {r["patient_id"] for r in parsed}
            for p in batch:
                if p["patient_id"] not in parsed_ids:
                    _db.upsert_patient(conn, {**p, "llm_raw_markdown": raw_response})

        n_ok += len(parsed)
        n_err += len(batch) - len(parsed)

    print(f"[pipeline] Done. {n_ok} patients processed successfully, {n_err} errors.")
    return n_ok, n_err
