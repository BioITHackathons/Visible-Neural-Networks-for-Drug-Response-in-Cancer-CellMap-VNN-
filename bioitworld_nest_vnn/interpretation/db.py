"""SQLite schema and CRUD helpers for NeST-VNN LLM interpretations."""

import json
import sqlite3
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent / "nest_vnn_interpretations.db"


def get_conn(db_path: Path = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path = DB_PATH) -> None:
    with get_conn(db_path) as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS patients (
            patient_id TEXT NOT NULL,
            study_id TEXT NOT NULL,
            label TEXT NOT NULL,
            predicted_score REAL,
            predicted_class TEXT,
            sample_type TEXT,
            mutation_count REAL,
            fraction_genome_altered REAL,
            llm_raw_markdown TEXT,
            processed_at TEXT,
            PRIMARY KEY (patient_id, study_id, label)
        );

        CREATE TABLE IF NOT EXISTS patient_nests (
            patient_id TEXT NOT NULL,
            study_id TEXT NOT NULL,
            label TEXT NOT NULL,
            nest_id TEXT NOT NULL,
            rank INTEGER,
            importance_score REAL,
            rlipp_score REAL,
            population_rlipp REAL,
            outcome_direction TEXT,
            pathway_name TEXT,
            reactome_link TEXT,
            biological_explanation TEXT,
            clinical_reasoning TEXT,
            PRIMARY KEY (patient_id, study_id, label, nest_id)
        );

        CREATE TABLE IF NOT EXISTS patient_genes (
            patient_id TEXT NOT NULL,
            study_id TEXT NOT NULL,
            label TEXT NOT NULL,
            nest_id TEXT NOT NULL,
            gene_name TEXT NOT NULL,
            rank INTEGER,
            importance_score REAL,
            alteration_type TEXT,
            outcome_direction TEXT,
            biological_role TEXT,
            drugs TEXT,
            PRIMARY KEY (patient_id, study_id, label, nest_id, gene_name)
        );
        """)


def upsert_patient(conn: sqlite3.Connection, rec: dict) -> None:
    conn.execute("""
        INSERT OR REPLACE INTO patients
        (patient_id, study_id, label, predicted_score, predicted_class,
         sample_type, mutation_count, fraction_genome_altered, llm_raw_markdown, processed_at)
        VALUES (?,?,?,?,?,?,?,?,?,?)
    """, (
        rec["patient_id"], rec["study_id"], rec["label"],
        rec.get("predicted_probability"), rec.get("predicted_class"),
        rec.get("sample_type"), rec.get("mutation_count"), rec.get("fraction_genome_altered"),
        rec.get("llm_raw_markdown"), datetime.utcnow().isoformat(),
    ))


def upsert_nest(conn: sqlite3.Connection, patient_id: str, study_id: str, label: str, nest: dict) -> None:
    conn.execute("""
        INSERT OR REPLACE INTO patient_nests
        (patient_id, study_id, label, nest_id, rank, importance_score, rlipp_score,
         population_rlipp, outcome_direction, pathway_name, reactome_link,
         biological_explanation, clinical_reasoning)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        patient_id, study_id, label,
        nest["nest_id"], nest.get("rank"), nest.get("importance_score"), nest.get("rlipp_score"),
        nest.get("population_rlipp"), nest.get("direction"),
        nest.get("pathway_name"), nest.get("reactome_link"),
        nest.get("biological_explanation"), nest.get("clinical_reasoning"),
    ))


def upsert_gene(conn: sqlite3.Connection, patient_id: str, study_id: str, label: str,
                nest_id: str, gene: dict) -> None:
    conn.execute("""
        INSERT OR REPLACE INTO patient_genes
        (patient_id, study_id, label, nest_id, gene_name, rank, importance_score,
         alteration_type, outcome_direction, biological_role, drugs)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)
    """, (
        patient_id, study_id, label, nest_id,
        gene["gene_name"], gene.get("rank"), gene.get("importance_score"),
        gene.get("alteration_type"), gene.get("direction"),
        gene.get("biological_role"),
        json.dumps(gene.get("drugs", [])),
    ))


# ── Read helpers ──────────────────────────────────────────────────────────────

def list_patients(study_id: str = None, label: str = None, pred_class: str = None,
                  db_path: Path = DB_PATH) -> list[dict]:
    filters, args = [], []
    if study_id:
        filters.append("study_id = ?"); args.append(study_id)
    if label:
        filters.append("label = ?"); args.append(label)
    if pred_class:
        filters.append("predicted_class = ?"); args.append(pred_class)
    where = ("WHERE " + " AND ".join(filters)) if filters else ""
    with get_conn(db_path) as conn:
        rows = conn.execute(f"SELECT * FROM patients {where} ORDER BY predicted_score DESC", args).fetchall()
    return [dict(r) for r in rows]


def get_patient(patient_id: str, study_id: str, label: str, db_path: Path = DB_PATH) -> dict | None:
    with get_conn(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM patients WHERE patient_id=? AND study_id=? AND label=?",
            (patient_id, study_id, label)
        ).fetchone()
    return dict(row) if row else None


def get_nests(patient_id: str, study_id: str, label: str, db_path: Path = DB_PATH) -> list[dict]:
    with get_conn(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM patient_nests WHERE patient_id=? AND study_id=? AND label=? ORDER BY rank",
            (patient_id, study_id, label)
        ).fetchall()
    return [dict(r) for r in rows]


def get_genes(patient_id: str, study_id: str, label: str, nest_id: str = None,
              db_path: Path = DB_PATH) -> list[dict]:
    if nest_id:
        with get_conn(db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM patient_genes WHERE patient_id=? AND study_id=? AND label=? AND nest_id=? ORDER BY rank",
                (patient_id, study_id, label, nest_id)
            ).fetchall()
    else:
        with get_conn(db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM patient_genes WHERE patient_id=? AND study_id=? AND label=? ORDER BY nest_id, rank",
                (patient_id, study_id, label)
            ).fetchall()
    return [dict(r) for r in rows]


def already_processed(patient_id: str, study_id: str, label: str,
                      db_path: Path = DB_PATH) -> bool:
    with get_conn(db_path) as conn:
        row = conn.execute(
            "SELECT llm_raw_markdown FROM patients WHERE patient_id=? AND study_id=? AND label=?",
            (patient_id, study_id, label)
        ).fetchone()
    return row is not None and row["llm_raw_markdown"] is not None
