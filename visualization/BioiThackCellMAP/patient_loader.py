"""
Parse patient_viz.html (NeST-VNN output) and expose per-patient data.
Loaded once at server startup; all functions read from in-memory cache.
"""
import re
import json
import os
import glob
from pathlib import Path

_data = None         # parsed data cache
_pat_index = {}      # patientId → array index (built on load)

# Input-matrix data (loaded from nest_vnn_input/ alongside patient_viz.html)
_gene_index = {}     # gene name → column index
_matrices   = {}     # 'mutation' | 'cnamplification' | 'cndeletion' | 'fusion' → list[bytearray]

def _build_search_paths():
    """Discover patient_viz.html under the sibling bioitworld_nest_vnn repo."""
    # vis/BioiThackCellMAP/ → cellmapvnn/ → bioitworld_nest_vnn/
    root = Path(__file__).resolve().parent.parent.parent / "bioitworld_nest_vnn"
    found = sorted(
        glob.glob(str(root / "data" / "output" / "**" / "annotation" / "patient_viz.html"), recursive=True)
        + glob.glob(str(root / "mlruns" / "**" / "artifacts" / "annotation" / "patient_viz.html"), recursive=True),
        key=os.path.getmtime,
        reverse=True,  # most recently modified first
    )
    return found

SEARCH_PATHS = _build_search_paths()


def _extract_js_val(js, name):
    m = re.search(rf'const {re.escape(name)}\s*=\s*', js)
    if not m:
        return None
    start = m.end()
    first = js[start]
    close = {'{': '}', '[': ']'}.get(first)
    if close is None:
        end = js.index(';', start)
        return js[start:end].strip().strip('"').strip("'")
    depth = 0
    for i in range(start, len(js)):
        c = js[i]
        if c == first:
            depth += 1
        elif c == close:
            depth -= 1
            if depth == 0:
                return json.loads(js[start:i + 1])
    return None


def _load_input_matrices(study_dir):
    """Load binary alteration matrices from nest_vnn_input/ under study_dir."""
    global _gene_index, _matrices
    input_dir = os.path.join(study_dir, 'nest_vnn_input')
    if not os.path.isdir(input_dir):
        print(f'patient_loader: no nest_vnn_input dir at {study_dir}')
        return

    g2i_path = os.path.join(input_dir, 'gene2ind.txt')
    if not os.path.exists(g2i_path):
        print(f'patient_loader: no gene2ind.txt at {input_dir}')
        return
    with open(g2i_path) as f:
        for line in f:
            parts = line.strip().split('\t', 1)
            if len(parts) == 2:
                _gene_index[parts[1]] = int(parts[0])

    for name in ('mutation', 'cnamplification', 'cndeletion', 'fusion'):
        path = os.path.join(input_dir, f'cell2{name}.txt')
        if not os.path.exists(path):
            continue
        rows = []
        with open(path) as f:
            for line in f:
                rows.append(bytearray(int(v) for v in line.strip().split(',')))
        _matrices[name] = rows
        print(f'patient_loader: loaded {name} '
              f'({len(rows)} rows × {len(rows[0]) if rows else 0} genes)')


def load(custom_path=None):
    global _data, _pat_index
    if _data is not None:
        return True

    candidates = ([custom_path] if custom_path else []) + SEARCH_PATHS
    chosen = next((p for p in candidates if p and os.path.exists(p)), None)
    if chosen is None:
        print('patient_loader: no patient_viz.html found')
        return False

    print(f'patient_loader: loading {chosen} …')
    with open(chosen, encoding='utf-8') as f:
        html = f.read()

    try:
        s = html.index('<script>')
        e = html.rindex('</script>')
        js = html[s + 8:e]
    except ValueError:
        print('patient_loader: no <script> block found')
        return False

    keys = ['cellIds', 'preds', 'termImp', 'ptRlipp', 'termMZ',
            'geneImp', 'geneSZ', 'popRlipp', 'popTermPrho', 'popGeneRho',
            'studyId', 'labelName', 'task']
    d = {}
    for k in keys:
        v = _extract_js_val(js, k)
        if v is None:
            print(f'patient_loader: WARNING — could not parse {k}')
        d[k] = v

    _data = d
    _pat_index = {pid: i for i, pid in enumerate(_data.get('cellIds') or [])}
    print(f'patient_loader: {len(_pat_index)} patients, '
          f'{len(_data.get("termImp") or {})} terms, '
          f'{len(_data.get("geneImp") or {})} genes')

    # patient_viz.html lives at {study_dir}/{label}/annotation/patient_viz.html
    study_dir = os.path.dirname(os.path.dirname(os.path.dirname(chosen)))
    _load_input_matrices(study_dir)
    return True


def is_loaded():
    return _data is not None


# ── Direction helpers (mirrors patient_viz.html JS logic) ─────────────────

def term_dir(term, pat_idx):
    """
    +1 → patient deviation pushes toward HIGHER predicted score
    -1 → pushes toward LOWER
     0 → signal too weak to call
    """
    mz_list = (_data.get('termMZ') or {}).get(term, [])
    mz = mz_list[pat_idx] if pat_idx < len(mz_list) else 0.0
    rho = (_data.get('popTermPrho') or {}).get(term, 0.0) or 0.0
    if abs(mz) < 0.2 or abs(rho) < 0.1:
        return 0
    return (1 if mz > 0 else -1) * (1 if rho > 0 else -1)


def gene_dir(gene, pat_idx):
    sz_list = (_data.get('geneSZ') or {}).get(gene, [])
    sz = sz_list[pat_idx] if pat_idx < len(sz_list) else 0.0
    rho = (_data.get('popGeneRho') or {}).get(gene, 0.0) or 0.0
    if abs(sz) < 0.5 or abs(rho) < 0.1:
        return 0
    return (1 if sz > 0 else -1) * (1 if rho > 0 else -1)


# ── RLIPP classification (mirrors patient_viz.html) ───────────────────────

def rlipp_tier(v):
    if v is None:
        return 'none'
    if v > 1.2:
        return 'high'
    if v > 1.0:
        return 'mid'
    return 'low'


# ── Public API ────────────────────────────────────────────────────────────

def get_patient_list():
    if not _data:
        return []
    cells = _data.get('cellIds') or []
    preds = _data.get('preds') or []
    return [
        {'id': pid, 'pred': round(preds[i], 4) if i < len(preds) else None}
        for i, pid in enumerate(cells)
    ]


def get_top_nests(patient_id, n=10):
    if not _data:
        return None
    pat_idx = _pat_index.get(patient_id)
    if pat_idx is None:
        return None

    ti = _data.get('termImp') or {}
    terms = []
    for term, imps in ti.items():
        imp = imps[pat_idx] if pat_idx < len(imps) else 0.0
        terms.append((term, imp))
    terms.sort(key=lambda x: x[1], reverse=True)

    result = []
    for term, imp in terms:
        d = term_dir(term, pat_idx)
        if d == 0:
            continue  # skip neutral — only return terms with a clear direction
        pt_r_list = (_data.get('ptRlipp') or {}).get(term, [])
        pt_r = pt_r_list[pat_idx] if pt_r_list and pat_idx < len(pt_r_list) else None
        pop_r = (_data.get('popRlipp') or {}).get(term)
        mz_list = (_data.get('termMZ') or {}).get(term, [])
        mz = mz_list[pat_idx] if pat_idx < len(mz_list) else None

        result.append({
            'nestId':   term,
            'imp':      round(imp, 4),
            'ptRlipp':  round(pt_r, 4) if pt_r is not None else None,
            'popRlipp': round(pop_r, 4) if pop_r is not None else None,
            'dir':      d,
            'mz':       round(mz, 4) if mz is not None else None,
        })
        if len(result) >= n:
            break
    return result


def get_gene_directions(patient_id, genes):
    if not _data:
        return None
    pat_idx = _pat_index.get(patient_id)
    if pat_idx is None:
        return None

    gi = _data.get('geneImp') or {}
    gs = _data.get('geneSZ') or {}
    gr = _data.get('popGeneRho') or {}

    out = {}
    for gene in genes:
        imp_list = gi.get(gene, [])
        sz_list  = gs.get(gene, [])
        imp = imp_list[pat_idx] if pat_idx < len(imp_list) else 0.0
        sz  = sz_list[pat_idx]  if pat_idx < len(sz_list)  else 0.0
        out[gene] = {
            'imp': round(imp, 4),
            'sz':  round(sz, 4),
            'rho': round(gr.get(gene, 0.0), 4),
            'dir': gene_dir(gene, pat_idx),
        }
    return out


def get_gene_alterations(patient_id, genes):
    """Return which alteration types are present for each gene in this patient."""
    if not _data:
        return None
    pat_idx = _pat_index.get(patient_id)
    if pat_idx is None:
        return None

    result = {}
    for gene in genes:
        col = _gene_index.get(gene)
        present = {}
        if col is not None:
            for name, rows in _matrices.items():
                if pat_idx < len(rows) and col < len(rows[pat_idx]):
                    if rows[pat_idx][col]:
                        present[name] = 1
        result[gene] = present
    return result


def get_meta():
    if not _data:
        return {}
    return {
        'study':     _data.get('studyId', ''),
        'label':     _data.get('labelName', ''),
        'task':      _data.get('task', ''),
        'nPatients': len(_pat_index),
        'nTerms':    len(_data.get('termImp') or {}),
        'nGenes':    len(_data.get('geneImp') or {}),
    }
