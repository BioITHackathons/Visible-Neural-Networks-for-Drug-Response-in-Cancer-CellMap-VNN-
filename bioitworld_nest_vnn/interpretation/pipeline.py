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

_PATIENT_HDR_RE = re.compile(r"##\s*Patient:\s*(.+?)[\r\n]")
_SECTION_RE = re.compile(r"####\s*(\d+)\.\s*(NEST:\d+)", re.IGNORECASE)
_PATHWAY_RE = re.compile(r"\*\*Pathway Name:\*\*\s*(.+)", re.IGNORECASE)
_REACTOME_RE = re.compile(r"\*\*Reactome Link:\*\*\s*(\S+)", re.IGNORECASE)
_BIO_EXP_RE = re.compile(r"\*\*Biological Explanation:\*\*\s*(.+?)(?=\*\*|\Z)", re.DOTALL | re.IGNORECASE)
_CLIN_RE = re.compile(r"\*\*Clinical Reasoning:\*\*\s*(.+?)(?=\*\*|\Z)", re.DOTALL | re.IGNORECASE)
_SUMMARY_RE = re.compile(r"\*\*Clinical Summary:\*\*\s*(.+?)(?=---|\Z)", re.DOTALL | re.IGNORECASE)
_TABLE_ROW_RE = re.compile(r"\|\s*([A-Z0-9]+)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|")


def _clean(s: str) -> str:
    return s.strip().replace("\n", " ").replace("  ", " ") if s else ""


def _parse_gene_table(block: str) -> list[dict]:
    genes = []
    for m in _TABLE_ROW_RE.finditer(block):
        gene, alt, direction, bio_role, drugs_raw = (m.group(i) for i in range(1, 6))
        if gene.lower() in ("gene", "---"):
            continue
        drug_list = [d.strip() for d in re.split(r"[,;]", drugs_raw) if d.strip() and d.strip().lower() != "none known"]
        genes.append({
            "gene_name": gene.strip(),
            "alteration_type": alt.strip(),
            "direction": direction.strip(),
            "biological_role": _clean(bio_role),
            "drugs": drug_list,
        })
    return genes


def _parse_llm_response(raw: str, patient_batch: list[dict]) -> list[dict]:
    """Parse LLM markdown response into structured per-patient dicts."""
    results = []
    patient_map = {p["patient_id"]: p for p in patient_batch}

    # Split raw text into one section per patient
    sections = re.split(r"(?=##\s*Patient:)", raw)

    for section in sections:
        hdr = _PATIENT_HDR_RE.match(section)
        if not hdr:
            continue
        patient_id = hdr.group(1).strip()
        block = section

        # Fuzzy fallback: if exact ID not found, try contains match
        if patient_id not in patient_map:
            patient_id = next(
                (pid for pid in patient_map if pid in patient_id or patient_id in pid),
                patient_id,
            )

        base = patient_map.get(patient_id, patient_batch[0])
        result = {
            "patient_id": patient_id,
            "study_id": base["study_id"],
            "label": base["label"],
            "predicted_probability": base.get("predicted_probability"),
            "predicted_class": base.get("predicted_class"),
            "sample_type": base.get("sample_type"),
            "mutation_count": base.get("mutation_count"),
            "fraction_genome_altered": base.get("fraction_genome_altered"),
            "llm_raw_markdown": block.strip(),
            "nests": [],
        }

        # Clinical summary
        sm = _SUMMARY_RE.search(block)
        result["clinical_summary"] = _clean(sm.group(1)) if sm else ""

        # Split block by NEST sections
        nest_positions = [(m.start(), m.group(2)) for m in _SECTION_RE.finditer(block)]
        for idx, (pos, nest_id) in enumerate(nest_positions):
            end_pos = nest_positions[idx + 1][0] if idx + 1 < len(nest_positions) else len(block)
            nest_block = block[pos:end_pos]

            pathway = _clean(m.group(1)) if (m := _PATHWAY_RE.search(nest_block)) else ""
            reactome = _clean(m.group(1)) if (m := _REACTOME_RE.search(nest_block)) else "N/A"
            bio_exp = _clean(m.group(1)) if (m := _BIO_EXP_RE.search(nest_block)) else ""
            clin = _clean(m.group(1)) if (m := _CLIN_RE.search(nest_block)) else ""

            # Match back to extracted nest data for importance/scores
            orig_nests = {n["nest_id"]: n for n in base.get("top_nests", [])}
            orig = orig_nests.get(nest_id, {})

            genes_parsed = _parse_gene_table(nest_block)
            # Merge importance from original extraction + replace drugs with CFDE data
            orig_genes = {g["gene_name"]: g for g in orig.get("top_patient_genes", [])}
            for g in genes_parsed:
                og = orig_genes.get(g["gene_name"], {})
                g["rank"] = og.get("rank", 0)
                g["importance_score"] = og.get("importance_score")
                # Replace LLM-suggested drugs with CFDE-verified approved drugs.
                # If the gene has no CFDE entry, keep whatever the LLM produced.
                cfde = _DRUG_LOOKUP.get(g["gene_name"].upper())
                if cfde is not None:
                    g["drugs"] = cfde

            result["nests"].append({
                "nest_id": nest_id,
                "rank": orig.get("rank", idx + 1),
                "importance_score": orig.get("importance_score"),
                "rlipp_score": orig.get("rlipp_score"),
                "population_rlipp": orig.get("population_rlipp"),
                "direction": orig.get("direction", ""),
                "pathway_name": pathway,
                "reactome_link": reactome if reactome.startswith("http") else "N/A",
                "biological_explanation": bio_exp,
                "clinical_reasoning": clin,
                "genes": genes_parsed,
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
