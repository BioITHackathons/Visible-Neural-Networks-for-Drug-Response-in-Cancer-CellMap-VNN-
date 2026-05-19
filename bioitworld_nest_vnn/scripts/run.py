"""
Interactive launcher for NeST-VNN train / predict / annotate.

Run from the repository root:
    python scripts/run.py
"""

import subprocess
import sys
from pathlib import Path

try:
    import questionary
    _Q = True
except ImportError:
    _Q = False

DATA_DIR = Path("data")
SRC_DIR = Path("src")
SCRIPTS_DIR = Path(__file__).parent

_SAMPLE_STUDY_ID = "nest_vnn_sample"
_SAMPLE_BASE_URL = "https://raw.githubusercontent.com/idekerlab/nest_vnn/main/sample"
_SAMPLE_FILES = [
    "training_data.txt",
    "test_data.txt",
    "cell2ind.txt",
    "gene2ind.txt",
    "ontology.txt",
    "cell2cnamplification.txt",
    "cell2cndeletion.txt",
    "cell2mutation.txt",
    "std.txt",
]

# Directories inside a study folder that are never label names
_NON_LABEL_DIRS = {"nest_vnn_input", "cbioportal_output"}


# ── Dataset / label discovery ─────────────────────────────────────────────────

def list_downloaded_studies() -> list[str]:
    """Studies that have cbioportal_output/mutations.csv ready for transformation."""
    output_dir = DATA_DIR / "output"
    if not output_dir.exists():
        return []
    return sorted(
        d.name for d in output_dir.iterdir()
        if d.is_dir() and (d / "cbioportal_output" / "mutations.csv").exists()
    )


def list_studies() -> list[str]:
    """Studies that have nest_vnn_input/training_data.txt ready."""
    output_dir = DATA_DIR / "output"
    if not output_dir.exists():
        return []
    return sorted(
        d.name for d in output_dir.iterdir()
        if d.is_dir() and (d / "nest_vnn_input" / "training_data.txt").exists()
    )


def list_labels(study_id: str) -> list[str]:
    """Label columns available in training_data.txt (skips cell_line / dataset)."""
    f = DATA_DIR / "output" / study_id / "nest_vnn_input" / "training_data.txt"
    if not f.exists():
        return []
    with open(f) as fh:
        header = fh.readline().strip()
    if "cell_line" not in header:
        return ["auc"]           # legacy 4-column format
    skip = {"cell_line", "dataset"}
    return [c for c in header.split("\t") if c not in skip]


def list_trained_labels(study_id: str) -> list[str]:
    """Labels for which a trained model_final.pt exists (new or old layout)."""
    study_dir = DATA_DIR / "output" / study_id
    found = []

    # New layout: <study>/<label>/model/model_final.pt
    for d in sorted(study_dir.iterdir()):
        if d.is_dir() and d.name not in _NON_LABEL_DIRS:
            if (d / "model" / "model_final.pt").exists():
                found.append(d.name)

    # Old layout: <study>/model/model_final.pt (no label in path)
    if not found and (study_dir / "model" / "model_final.pt").exists():
        # Use training_data header to surface available label names
        found = list_labels(study_id)

    return found


def list_predicted_labels(study_id: str) -> list[str]:
    """Labels for which predict.txt exists (new or old layout)."""
    study_dir = DATA_DIR / "output" / study_id
    found = []

    # New layout
    for d in sorted(study_dir.iterdir()):
        if d.is_dir() and d.name not in _NON_LABEL_DIRS:
            if (d / "metrics" / "predict.txt").exists():
                found.append(d.name)

    # Old layout
    if not found and (study_dir / "metrics" / "predict.txt").exists():
        found = list_labels(study_id)

    return found


# ── Path helpers (support both old and new layouts) ───────────────────────────

def _nest_dir(study_id: str) -> Path:
    return DATA_DIR / "output" / study_id / "nest_vnn_input"


def _model_dir(study_id: str, label: str) -> Path:
    """Return model directory, preferring new layout over old."""
    new = DATA_DIR / "output" / study_id / label / "model"
    if new.exists():
        return new
    old = DATA_DIR / "output" / study_id / "model"
    if old.exists():
        return old
    return new                   # will be created by training


def _metrics_dir(study_id: str, label: str) -> Path:
    new = DATA_DIR / "output" / study_id / label / "metrics"
    if new.exists():
        return new
    old = DATA_DIR / "output" / study_id / "metrics"
    if old.exists():
        return old
    return new


# ── Interactive helpers ───────────────────────────────────────────────────────

def _select(prompt: str, options: list[str]) -> str:
    if not options:
        print(f"\nNo options available: {prompt}")
        sys.exit(1)
    if _Q:
        result = questionary.select(prompt, choices=options).ask()
        if result is None:
            sys.exit(0)
        return result
    print(f"\n{prompt}")
    for i, opt in enumerate(options):
        print(f"  [{i}] {opt}")
    while True:
        raw = input("> ").strip()
        try:
            idx = int(raw)
            if 0 <= idx < len(options):
                return options[idx]
        except ValueError:
            pass
        print(f"  Enter a number 0–{len(options) - 1}.")


def _param(name: str, default, hint: str = "") -> str:
    """Prompt for one parameter; press Enter to keep the default."""
    if _Q:
        suffix = f" ({hint})" if hint else ""
        result = questionary.text(f"{name}{suffix}:", default=str(default)).ask()
        if result is None:
            sys.exit(0)
        return result or str(default)
    suffix = f"  ({hint})" if hint else ""
    raw = input(f"  {name} [{default}]{suffix}: ").strip()
    return raw if raw else str(default)


def _confirm(prompt: str, default: bool = False) -> bool:
    """Yes/no prompt."""
    if _Q:
        result = questionary.confirm(prompt, default=default).ask()
        if result is None:
            sys.exit(0)
        return result
    raw = input(f"{prompt} [{'Y/n' if default else 'y/N'}]: ").strip().lower()
    if not raw:
        return default
    return raw == "y"


def _select_zscore(default: str = "auc") -> str:
    choices = ["auc", "zscore", "robustz"]
    if _Q:
        result = questionary.select(
            "zscore_method (auc = no normalization):",
            choices=choices,
            default=default,
        ).ask()
        if result is None:
            sys.exit(0)
        return result
    print(f"  zscore_method [{default}]  (auc/zscore/robustz): ", end="")
    raw = input().strip()
    return raw if raw in choices else default


def _infer_task(label: str) -> str:
    return "binary" if "binary" in label.lower() else "continuous"


def _resolve_task(study_id: str, label: str) -> tuple[str, str]:
    """Return (task, source) from endpoints.json; fall back to label-name inference."""
    import json
    ep_path = DATA_DIR / "output" / study_id / "nest_vnn_input" / "endpoints.json"
    if ep_path.exists():
        try:
            endpoints = json.loads(ep_path.read_text())
            for ep in endpoints:
                if ep.get("label_name") == label:
                    task = "binary" if "binary" in ep.get("mode", "") else "continuous"
                    return task, "endpoints.json"
        except Exception:
            pass
    return _infer_task(label), "label name"


def _header(title: str, **fields):
    print(f"\n{'='*60}")
    print(title)
    for k, v in fields.items():
        print(f"  {k+':':10s} {v}")
    print("=" * 60)


# ── Mode implementations ──────────────────────────────────────────────────────

def run_train():
    studies = list_studies()
    if not studies:
        print("\nNo studies ready. Run cbioport_transform.py first.")
        return

    study_id = _select("Select study:", studies)

    labels = list_labels(study_id)
    if not labels:
        print(f"\nNo label columns found in training_data.txt for '{study_id}'.")
        return
    label = _select("Select target variable:", labels)

    ndir  = _nest_dir(study_id)
    mdir  = DATA_DIR / "output" / study_id / label / "model"   # always new layout

    task, task_source = _resolve_task(study_id, label)

    print("\nParameters (press Enter to keep default):")
    print(f"  task:       {task}  (from {task_source})")
    cuda             = _param("cuda",             "0",      "GPU index or 'cpu'")
    epochs           = _param("epoch",            "200")
    batchsize        = _param("batchsize",         "512")
    lr               = _param("lr",               "0.001", "learning rate")
    optimize         = _param("optimize",          "1",      "1=direct  2=Optuna search")
    genotype_hiddens = _param("genotype_hiddens",  "4",      "hidden units per ontology term")
    zscore_method    = _select_zscore("auc")

    print("\nAdvanced parameters (press Enter to keep default):")
    wd               = _param("wd",               "0.001",  "weight decay")
    alpha            = _param("alpha",             "0.3",    "auxiliary loss weight")
    patience         = _param("patience",          "30",     "early stopping patience (epochs)")
    dropout_fraction  = _param("dropout_fraction",  "0.3",    "dropout fraction")
    min_dropout_layer = _param("min_dropout_layer", "2",      "first ontology layer to apply dropout")
    seed_raw          = _param("seed",              "",       "random seed (leave blank for none)")

    use_mlflow = _confirm("Enable MLflow tracking?", default=True)

    mdir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(SRC_DIR / "train.py"),
        "-onto",              str(ndir / "ontology.txt"),
        "-gene2id",           str(ndir / "gene2ind.txt"),
        "-cell2id",           str(ndir / "cell2ind.txt"),
        "-train",             str(ndir / "training_data.txt"),
        "-mutations",         str(ndir / "cell2mutation.txt"),
        "-cn_deletions",      str(ndir / "cell2cndeletion.txt"),
        "-cn_amplifications", str(ndir / "cell2cnamplification.txt"),
        "-label",             label,
        "-task",              task,
        "-std",               str(mdir / "std.txt"),
        "-model",             str(mdir),
        "-genotype_hiddens",  genotype_hiddens,
        "-lr",                lr,
        "-wd",                wd,
        "-alpha",             alpha,
        "-cuda",              cuda,
        "-epoch",             epochs,
        "-batchsize",         batchsize,
        "-optimize",          optimize,
        "-zscore_method",     zscore_method,
        "-patience",          patience,
        "-dropout_fraction",  dropout_fraction,
        "-min_dropout_layer", min_dropout_layer,
    ]
    if seed_raw.strip():
        cmd += ["-seed", seed_raw.strip()]
    if use_mlflow:
        cmd += ["-mlflow"]
    fusion_file = ndir / "cell2fusion.txt"
    if fusion_file.exists():
        cmd += ["-fusions", str(fusion_file)]

    _header("Training NeST-VNN", Study=study_id, Label=label, Task=task, Model=str(mdir))
    subprocess.run(cmd, check=True)


def run_predict():
    studies = list_studies()
    if not studies:
        print("\nNo studies ready.")
        return

    study_id = _select("Select study:", studies)

    labels = list_trained_labels(study_id)
    if not labels:
        print(f"\nNo trained models found for '{study_id}'. Run train first.")
        return
    label = _select("Select label (trained run):", labels)

    ndir    = _nest_dir(study_id)
    mdir    = _model_dir(study_id, label)
    metrdir = DATA_DIR / "output" / study_id / label / "metrics"   # always new layout

    task, task_source = _resolve_task(study_id, label)

    print("\nParameters (press Enter to keep default):")
    print(f"  task:       {task}  (from {task_source})")
    cuda       = _param("cuda",      "0",  "GPU index")
    batchsize  = _param("batchsize", "64")
    use_mlflow = _confirm("Enable MLflow tracking?", default=True)

    (metrdir / "hidden").mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(SRC_DIR / "predict.py"),
        "-predict",           str(ndir / "training_data.txt"),
        "-gene2id",           str(ndir / "gene2ind.txt"),
        "-cell2id",           str(ndir / "cell2ind.txt"),
        "-mutations",         str(ndir / "cell2mutation.txt"),
        "-cn_deletions",      str(ndir / "cell2cndeletion.txt"),
        "-cn_amplifications", str(ndir / "cell2cnamplification.txt"),
        "-label",             label,
        "-task",              task,
        "-std",               str(mdir / "std.txt"),
        "-load",              str(mdir / "model_final.pt"),
        "-hidden",            str(metrdir / "hidden"),
        "-result",            str(metrdir / "predict"),
        "-cuda",              cuda,
        "-batchsize",         batchsize,
    ]
    if use_mlflow:
        cmd += ["-mlflow"]
    fusion_file = ndir / "cell2fusion.txt"
    if fusion_file.exists():
        cmd += ["-fusions", str(fusion_file)]

    _header("Predicting with NeST-VNN",
            Study=study_id, Label=label, Task=task,
            Model=str(mdir / "model_final.pt"), Output=str(metrdir))
    subprocess.run(cmd, check=True)


def run_predict_annotate():
    studies = list_studies()
    if not studies:
        print("\nNo studies ready.")
        return

    study_id = _select("Select study:", studies)

    labels = list_trained_labels(study_id)
    if not labels:
        print(f"\nNo trained models found for '{study_id}'. Run train first.")
        return
    label = _select("Select label (trained run):", labels)

    ndir    = _nest_dir(study_id)
    mdir    = _model_dir(study_id, label)
    metrdir = DATA_DIR / "output" / study_id / label / "metrics"

    task, task_source = _resolve_task(study_id, label)

    print("\nParameters (press Enter to keep default):")
    print(f"  task:       {task}  (from {task_source})")
    cuda       = _param("cuda",      "0",  "GPU index")
    batchsize  = _param("batchsize", "64")
    cpu_count  = _param("cpu_count", "4",  "parallel CPUs for RLIPP scoring")
    use_mlflow = _confirm("Enable MLflow tracking?", default=True)

    (metrdir / "hidden").mkdir(parents=True, exist_ok=True)

    predict_cmd = [
        sys.executable, str(SRC_DIR / "predict.py"),
        "-predict",           str(ndir / "training_data.txt"),
        "-gene2id",           str(ndir / "gene2ind.txt"),
        "-cell2id",           str(ndir / "cell2ind.txt"),
        "-mutations",         str(ndir / "cell2mutation.txt"),
        "-cn_deletions",      str(ndir / "cell2cndeletion.txt"),
        "-cn_amplifications", str(ndir / "cell2cnamplification.txt"),
        "-label",             label,
        "-task",              task,
        "-std",               str(mdir / "std.txt"),
        "-load",              str(mdir / "model_final.pt"),
        "-hidden",            str(metrdir / "hidden"),
        "-result",            str(metrdir / "predict"),
        "-cuda",              cuda,
        "-batchsize",         batchsize,
    ]
    if use_mlflow:
        predict_cmd += ["-mlflow"]
    fusion_file = ndir / "cell2fusion.txt"
    if fusion_file.exists():
        predict_cmd += ["-fusions", str(fusion_file)]

    annotate_cmd = [
        sys.executable, str(SRC_DIR / "annotate_hierarchy.py"),
        study_id,
        "-label",   label,
        "-task",    task,
        "-cpu_count", cpu_count,
    ]
    if use_mlflow:
        annotate_cmd += ["-mlflow"]

    _header("Predict + Annotate NeST-VNN",
            Study=study_id, Label=label, Task=task,
            Model=str(mdir / "model_final.pt"), Output=str(metrdir))
    subprocess.run(predict_cmd, check=True)
    subprocess.run(annotate_cmd, check=True)


def run_annotate():
    studies = list_studies()
    if not studies:
        print("\nNo studies ready.")
        return

    study_id = _select("Select study:", studies)

    labels = list_predicted_labels(study_id)
    if not labels:
        print(f"\nNo predictions found for '{study_id}'. Run predict first.")
        return
    label = _select("Select label (predicted run):", labels)

    task, task_source = _resolve_task(study_id, label)
    print("\nParameters (press Enter to keep default):")
    print(f"  task:       {task}  (from {task_source})")
    cpu_count = _param("cpu_count", "4", "parallel CPUs for RLIPP scoring")

    cmd = [
        sys.executable, str(SRC_DIR / "annotate_hierarchy.py"),
        study_id,
        "-label",     label,
        "-task",      task,
        "-cpu_count", cpu_count,
    ]

    _header("Annotating NeST-VNN Hierarchy",
            Study=study_id, Label=label, Task=task)
    subprocess.run(cmd, check=True)


# ── cBioPortal download / transform ──────────────────────────────────────────

def run_download():
    """Download genomic + clinical data for a cBioPortal study."""
    if _Q:
        study_id = questionary.text("cBioPortal study ID (e.g. laml_tcga_pub, breast_msk_2025):").ask()
        if not study_id:
            return
    else:
        print("\nEnter a cBioPortal study ID (e.g. laml_tcga_pub, breast_msk_2025):")
        study_id = input("  study_id: ").strip()
        if not study_id:
            print("No study ID entered.")
            return

    _header("Downloading from cBioPortal", Study=study_id,
            Output=str(DATA_DIR / "output" / study_id / "cbioportal_output"))
    cmd = [sys.executable, str(SCRIPTS_DIR / "cbioport_download.py"), study_id]
    subprocess.run(cmd, check=True)


def run_transform():
    """Transform downloaded cBioPortal data into NeST-VNN input files."""
    studies = list_downloaded_studies()
    if not studies:
        print("\nNo downloaded studies found. Run 'Download from cBioPortal' first.")
        return

    study_id = _select("Select study to transform:", studies)

    _header("Transforming to NeST-VNN Format", Study=study_id,
            Output=str(DATA_DIR / "output" / study_id / "nest_vnn_input"))
    cmd = [sys.executable, str(SCRIPTS_DIR / "cbioport_transform.py"), study_id]
    subprocess.run(cmd, check=True)


# ── Sample data download ──────────────────────────────────────────────────────

def run_download_sample():
    """Download the original NeST-VNN sample data (GDSC drug-response) and set it up."""
    import requests as _req

    output_dir = DATA_DIR / "output" / _SAMPLE_STUDY_ID / "nest_vnn_input"

    existing = [f for f in _SAMPLE_FILES if (output_dir / f).exists()]
    if existing:
        print(f"\nSample data already present in {output_dir}/")
        print(f"  {len(existing)}/{len(_SAMPLE_FILES)} files found.")
        if not _confirm("Re-download?"):
            _sample_usage(output_dir)
            return

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nDownloading NeST-VNN sample data → {output_dir}/")
    print(f"  Source: {_SAMPLE_BASE_URL}\n")

    failed = []
    for fname in _SAMPLE_FILES:
        url = f"{_SAMPLE_BASE_URL}/{fname}"
        dest = output_dir / fname
        try:
            resp = _req.get(url, timeout=60)
            resp.raise_for_status()
            dest.write_bytes(resp.content)
            print(f"  ✓ {fname:<35s} ({len(resp.content)/1024:.0f} KB)")
        except Exception as e:
            print(f"  ✗ {fname:<35s} {e}")
            failed.append(fname)

    if failed:
        print(f"\n⚠ {len(failed)} file(s) failed: {failed}")
        print(f"  Download manually from {_SAMPLE_BASE_URL}")
        return

    _sample_usage(output_dir)


def _sample_usage(output_dir: Path):
    print(f"\n{'='*60}")
    print(f"Sample data ready: {output_dir.resolve()}")
    print(f"{'='*60}")
    print(f"""
Dataset: GDSC cancer cell line drug-response (Sanger/Broad)
  Format : 4-column TSV — cell_line | SMILES | AUC | dataset
  Cells  : 1 244 cell lines   Genes: 718   Ontology: NeST
  Label  : auc  (continuous regression, zscore_method=auc)

Quick start — select these when prompted:
  python scripts/run.py
    → Train → {_SAMPLE_STUDY_ID} → auc → continuous

For prediction on the held-out test set, the test_data.txt file
has been saved alongside training_data.txt. Use predict.py with
  -predict data/output/{_SAMPLE_STUDY_ID}/nest_vnn_input/test_data.txt
""")
    if _confirm("Launch training now?"):
        run_train()


# ── Entry point ───────────────────────────────────────────────────────────────

_MODES = {
    "Download from cBioPortal": run_download,
    "Transform to NeST-VNN":    run_transform,
    "Train":                    run_train,
    "Predict + Annotate":       run_predict_annotate,
    "Predict":                  run_predict,
    "Annotate":                 run_annotate,
    "Download sample data":     run_download_sample,
}


def main():
    mode = _select("What would you like to do?", list(_MODES.keys()))
    _MODES[mode]()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        import sys
        sys.exit(0)
