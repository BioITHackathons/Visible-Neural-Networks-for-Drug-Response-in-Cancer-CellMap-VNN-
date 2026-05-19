import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as du
from contextlib import nullcontext
from torch.autograd import Variable

import util
from training_data_wrapper import *
from drugcell_nn import *


def _render_nn_schematic(root, num_terms, num_genes, num_features, num_hiddens, task, out_path):
    """
    Render a conceptual schematic of the NeST-VNN architecture.

    Shows: gene input features → per-gene feature layers (sparse, ontology-guided) →
    leaf system hidden states → parent system hidden states → root → prediction,
    with auxiliary supervision heads on every term.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyBboxPatch

    W, H = 22.0, 11.0
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis('off')
    BG = '#f5f6fa'
    ax.set_facecolor(BG)
    fig.patch.set_facecolor(BG)

    # ── palette ───────────────────────────────────────────────────────────────
    CG_IN  = '#1e8449'   # gene input
    CG_LY  = '#27ae60'   # gene feature layer
    C_LEAF = '#2471a3'   # leaf system term
    C_MID  = '#1a5276'   # mid-level term
    C_ROOT = '#c0392b'   # root term
    C_AUX  = '#e67e22'   # auxiliary head
    C_PRED = '#7d3c98'   # final prediction
    C_SPR  = '#b0bec5'   # sparse connection
    C_DNS  = '#546e7a'   # dense connection

    # ── helpers ───────────────────────────────────────────────────────────────
    def box(x, y, w, h, color, label, sub=None, fs=8, tc='white', alpha=0.92):
        p = FancyBboxPatch((x - w/2, y - h/2), w, h,
                           boxstyle='round,pad=0.1', facecolor=color,
                           edgecolor='white', linewidth=1.8, alpha=alpha, zorder=3)
        ax.add_patch(p)
        ax.text(x, y + (0.13 if sub else 0), label,
                ha='center', va='center', fontsize=fs, fontweight='bold',
                color=tc, zorder=4, multialignment='center')
        if sub:
            ax.text(x, y - 0.22, sub,
                    ha='center', va='center', fontsize=fs - 1.5,
                    color=tc, alpha=0.87, zorder=4, style='italic',
                    multialignment='center')

    def arr(x1, y1, x2, y2, color='#555', lw=1.6, dashed=False):
        ls = (0, (5, 4)) if dashed else 'solid'
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                    linestyle=ls, mutation_scale=14), zorder=2)

    def hdr(x, y, txt):
        ax.text(x, y, txt, ha='center', va='top', fontsize=8.5,
                fontweight='bold', color='#1a252f', multialignment='center',
                bbox=dict(facecolor='white', edgecolor='#ccc',
                          boxstyle='round,pad=0.3', alpha=0.9))

    # ── column x-positions ────────────────────────────────────────────────────
    XGI  = 1.8    # gene inputs
    XGL  = 4.8    # gene feature layers
    XLEF = 8.3    # leaf terms
    XMID = 12.0   # parent terms
    XROO = 15.8   # root
    XPRD = 19.5   # prediction

    # ── section headers ───────────────────────────────────────────────────────
    for x, txt in [
        (XGI,  "Genomic\nInputs"),
        (XGL,  "Gene Feature\nLayers"),
        (XLEF, "Leaf System\nHidden States"),
        (XMID, "Parent System\nHidden States"),
        (XROO, "Root System\nHidden State"),
        (XPRD, "Clinical\nPrediction"),
    ]:
        hdr(x, 10.75, txt)

    # ── gene input boxes ──────────────────────────────────────────────────────
    feat_sub = "mut · cnv_del · cnv_amp" + (" · fusion" if num_features >= 4 else "")
    GYS = [8.6, 6.5, 4.4]   # y-positions for 3 example genes
    for i, y in enumerate(GYS):
        lbl = f"Gene {i + 1}" if i < 2 else f"Gene {num_genes}"
        box(XGI, y, 2.7, 0.82, CG_IN, lbl, sub=feat_sub, fs=7.5)
    ax.text(XGI, 5.5, '⋮', fontsize=22, ha='center', va='center', color='#999', zorder=4)
    ax.text(XGI, 3.3, f'×{num_genes} genes total', ha='center', fontsize=7,
            color='#666', style='italic')

    # ── gene feature layers ───────────────────────────────────────────────────
    for y in GYS:
        box(XGL, y, 2.5, 0.75, CG_LY, "Linear → tanh → BN", sub="scalar output", fs=7)
        arr(XGI + 1.35, y, XGL - 1.25, y, color=CG_IN, lw=1.5)
    ax.text(XGL, 5.5, '⋮', fontsize=22, ha='center', va='center', color='#999', zorder=4)

    # ── leaf terms ────────────────────────────────────────────────────────────
    YLA, YLB = 8.2, 5.2
    for y, lbl in [(YLA, "System A  (leaf)"), (YLB, "System B  (leaf)")]:
        box(XLEF, y, 3.0, 0.9, C_LEAF, lbl,
            sub=f"Linear → tanh → BN\nh: n_samples × {num_hiddens}", fs=7.5)

    ax.text(XLEF, 3.4, '⋮  (more leaf systems)', fontsize=8,
            ha='center', va='center', color='#999', style='italic')

    # sparse connections: genes 1+2 → System A, gene 3 → System B
    for gy in [8.6, 6.5]:
        arr(XGL + 1.25, gy, XLEF - 1.5, YLA, color=C_SPR, lw=1.0, dashed=True)
    arr(XGL + 1.25, 4.4, XLEF - 1.5, YLB, color=C_SPR, lw=1.0, dashed=True)
    ax.text((XGL + XLEF) / 2 - 0.3, 7.55,
            "ontology-guided\nsparse connections",
            ha='center', fontsize=6.5, color='#909090', style='italic')

    # aux heads on leaf terms
    for y, side, ay in [(YLA, +1, YLA + 1.45), (YLB, -1, YLB - 1.35)]:
        box(XLEF, ay, 2.4, 0.62, C_AUX, "Aux head",
            sub="Linear → tanh → Linear", fs=6.5)
        arr(XLEF, y + side * 0.45, XLEF, ay - side * 0.31, color=C_AUX, lw=1.1)

    # ── parent term ───────────────────────────────────────────────────────────
    YMID = 6.7
    box(XMID, YMID, 3.4, 1.05, C_MID,
        "Parent System",
        sub=f"input: [h_A ‖ h_B ‖ direct genes]\nLinear → tanh → BN  ·  h: n × {num_hiddens}", fs=7.5)

    arr(XLEF + 1.5, YLA, XMID - 1.7, YMID + 0.3, color=C_DNS, lw=2.0)
    arr(XLEF + 1.5, YLB, XMID - 1.7, YMID - 0.3, color=C_DNS, lw=2.0)
    ax.text((XLEF + XMID) / 2, YMID + 1.2,
            "dense\n(child hidden states)",
            ha='center', fontsize=6.5, color=C_DNS, style='italic')

    # aux head on parent term
    YAUXM = YMID + 1.8
    box(XMID, YAUXM, 2.4, 0.62, C_AUX, "Aux head",
        sub="Linear → tanh → Linear", fs=6.5)
    arr(XMID, YMID + 0.53, XMID, YAUXM - 0.31, color=C_AUX, lw=1.1)
    ax.text(XMID, YMID - 1.5, '⋮  (more system levels)',
            fontsize=8, ha='center', va='center', color='#999', style='italic')

    # ── root term ─────────────────────────────────────────────────────────────
    YROO = 6.7
    box(XROO, YROO, 3.4, 1.05, C_ROOT,
        f"Root  ({root})",
        sub=f"input: [h_children ‖ direct genes]\nLinear → tanh → BN  ·  h: n × {num_hiddens}", fs=7.5)

    # arrow from parent → root (with ellipsis notation for intermediate levels)
    arr(XMID + 1.7, YMID, XROO - 1.7, YROO, color=C_DNS, lw=2.0)
    ax.text((XMID + XROO) / 2, YROO + 0.35, '···',
            fontsize=18, ha='center', va='center', color='#bbb', zorder=5)

    # aux head on root
    YAUXR = YROO + 1.8
    box(XROO, YAUXR, 2.4, 0.62, C_AUX, "Aux head",
        sub="Linear → tanh → Linear", fs=6.5)
    arr(XROO, YROO + 0.53, XROO, YAUXR - 0.31, color=C_AUX, lw=1.1)

    # ── final prediction ──────────────────────────────────────────────────────
    sig = "\n[sigmoid → P(event)]" if task == 'binary' else ""
    box(XPRD, YROO, 3.2, 1.2, C_PRED,
        "Final Output",
        sub=f"Linear(h_root → 1){sig}\n→ predicted outcome", fs=7.5)
    arr(XROO + 1.7, YROO, XPRD - 1.6, YROO, color=C_PRED, lw=2.5)

    # ── callout boxes ─────────────────────────────────────────────────────────
    loss_fn = "BCEWithLogitsLoss" if task == 'binary' else "MSELoss"
    ax.text(XPRD, YROO - 1.6,
            f"Main loss:\n{loss_fn}",
            ha='center', va='top', fontsize=7.5, color='#4a235a',
            bbox=dict(facecolor='#f5eef8', edgecolor='#9b59b6',
                      boxstyle='round,pad=0.35', alpha=0.9))

    ax.text(8.5, 2.0,
            f"Aux loss (all terms, α = 0.3):\n"
            f"total_loss = main_loss + α · Σ_terms aux_loss\n"
            f"Propagates gradients to every level of the hierarchy",
            ha='center', va='top', fontsize=7, color='#784212',
            bbox=dict(facecolor='#fef9e7', edgecolor=C_AUX,
                      boxstyle='round,pad=0.4', alpha=0.9))

    # ── inset: inside one system term ─────────────────────────────────────────
    IX, IY, IW, IH = 12.1, 0.35, 9.6, 2.3
    ax.add_patch(FancyBboxPatch((IX, IY), IW, IH,
                                boxstyle='round,pad=0.12', facecolor='white',
                                edgecolor='#aaa', linewidth=1.2, alpha=0.97, zorder=2))
    ax.text(IX + IW / 2, IY + IH - 0.1,
            "Inside each system term",
            ha='center', va='top', fontsize=8, fontweight='bold',
            color='#1a252f', zorder=5)

    CY = IY + IH / 2 - 0.1
    comp_boxes = [
        (IX + 1.1,  CY, 1.9, 0.65, '#5d6d7e', "Concatenate",    "[child_h ‖ direct_gene_scalars]"),
        (IX + 3.3,  CY, 1.3, 0.65, C_MID,     "Linear",         f"→ {num_hiddens}D"),
        (IX + 4.95, CY, 1.1, 0.65, C_MID,     "tanh",           None),
        (IX + 6.35, CY, 1.5, 0.65, C_MID,     "BatchNorm",      f"{num_hiddens}D"),
        (IX + 8.1,  CY, 0.9, 0.65, C_LEAF,    "h",              f"{num_hiddens}D"),
    ]
    for bx, by, bw, bh, bc, bl, bsub in comp_boxes:
        box(bx, by, bw, bh, bc, bl, sub=bsub, fs=7)
    for x1, x2 in [(IX+2.05, IX+2.65), (IX+3.95, IX+4.4),
                   (IX+5.5,  IX+5.8),  (IX+7.1,  IX+7.65)]:
        arr(x1, CY, x2, CY, color='#555', lw=1.2)

    # ── legend ────────────────────────────────────────────────────────────────
    legend_items = [
        mpatches.Patch(color=CG_IN,  label=f'Gene input  ({feat_sub})'),
        mpatches.Patch(color=CG_LY,  label='Gene feature layer  (per-gene Linear → tanh → BN → scalar)'),
        mpatches.Patch(color=C_LEAF, label='Leaf system term  (hidden state)'),
        mpatches.Patch(color=C_MID,  label='Intermediate / parent system term'),
        mpatches.Patch(color=C_ROOT, label='Root system term'),
        mpatches.Patch(color=C_AUX,  label='Auxiliary output head  (per term, trained jointly)'),
        mpatches.Patch(color=C_PRED, label='Final prediction head'),
        mpatches.Patch(color=C_SPR,  label='Sparse connection  (ontology-guided mask: gene → term)'),
        mpatches.Patch(color=C_DNS,  label='Dense connection  (child hidden state → parent)'),
    ]
    leg = ax.legend(handles=legend_items, loc='lower left', fontsize=6.8,
                    framealpha=0.95, title='Legend', title_fontsize=7.5,
                    ncol=3, bbox_to_anchor=(0.0, 0.0),
                    borderpad=0.8, labelspacing=0.45)
    leg.get_frame().set_edgecolor('#ccc')

    ax.text(W / 2, 10.98,
            f"NeST-VNN Architecture — Conceptual Schematic  ·  "
            f"{num_terms} systems  ·  {num_genes} genes  ·  root: {root}",
            ha='center', va='top', fontsize=11, fontweight='bold', color='#1a252f')

    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close(fig)


def _render_nn_architecture(dG, term_size_map, term_direct_gene_map, root, out_path):
    """Render the ontology DAG (= NN architecture) as a PNG and save to out_path."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import networkx as nx

    # Assign each node a layer via longest-path from root (top-down BFS)
    layer = {root: 0}
    for node in nx.topological_sort(dG):
        for child in dG.successors(node):
            layer[child] = max(layer.get(child, 0), layer[node] + 1)

    max_layer = max(layer.values()) if layer else 0
    nodes_by_layer = {}
    for node, lyr in layer.items():
        nodes_by_layer.setdefault(lyr, []).append(node)

    # Assign x positions within each layer
    pos = {}
    for lyr, nodes in nodes_by_layer.items():
        nodes_sorted = sorted(nodes)
        width = len(nodes_sorted)
        for i, node in enumerate(nodes_sorted):
            pos[node] = ((i - (width - 1) / 2.0), -lyr)

    # Node color by number of directly annotated genes
    direct_counts = [len(term_direct_gene_map.get(n, [])) for n in dG.nodes()]
    max_count = max(direct_counts) if direct_counts else 1
    max_count = max_count or 1
    node_colors = [cm.YlOrRd(len(term_direct_gene_map.get(n, [])) / max_count) for n in dG.nodes()]

    # Node size by total annotated genes (term_size_map)
    max_size = max(term_size_map.values()) if term_size_map else 1
    node_sizes = [20 + 180 * (term_size_map.get(n, 1) / max_size) for n in dG.nodes()]

    num_nodes = len(dG.nodes())
    fig_w = max(12, num_nodes * 0.15)
    fig_h = max(6, (max_layer + 1) * 0.8)
    fig, ax = plt.subplots(figsize=(min(fig_w, 40), min(fig_h, 24)))

    nx.draw(
        dG, pos=pos, ax=ax,
        node_color=node_colors, node_size=node_sizes,
        edge_color="#aaaaaa", arrows=True, arrowsize=6,
        with_labels=False, alpha=0.85,
    )

    sm = plt.cm.ScalarMappable(cmap=cm.YlOrRd, norm=plt.Normalize(0, max_count))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, shrink=0.5, label="Direct gene annotations")

    ax.set_title(
        f"NeST-VNN Architecture  |  {num_nodes} terms  |  root: {root}",
        fontsize=11, pad=8,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


_HEADER_REPEAT = 20   # reprint column headers every N epochs


def _print_epoch_header(task):
    """Print a fixed-width column header with separator lines."""
    if task == 'binary':
        hdr = (f"{'Epoch':>5}  {'Train Acc':>9}  {'Train Loss':>10}  "
               f"{'Val Acc':>7}  {'Val Loss':>8}  {'Grad Norm':>9}  {'Time (s)':>8}")
    else:
        hdr = (f"{'Epoch':>5}  {'Train ρ':>7}  {'Train Loss':>10}  "
               f"{'Lbl μ':>5}  {'Pred μ':>6}  {'Val ρ':>5}  "
               f"{'Val Loss':>8}  {'Grad Norm':>9}  {'Time (s)':>8}")
    sep = '─' * len(hdr)
    print(f"{sep}\n{hdr}\n{sep}")


def _fmt_epoch_row(task, epoch, train_metric, train_loss, val_metric, val_loss,
                   gradnorms, elapsed, saved=False, true_auc=None, pred_auc=None):
    """Return a fixed-width formatted epoch row, with ✓ marker when model is saved."""
    def g(v, w, d=4):
        if v != v:  # NaN != NaN is True
            return f"{'nan':>{w}}"
        return f"{v:>{w}.{d}f}"

    if task == 'binary':
        row = (f"{epoch:>5}  {g(train_metric,9)}  {g(train_loss,10)}  "
               f"{g(val_metric,7)}  {g(val_loss,8)}  {g(gradnorms,9)}  {g(elapsed,8,1)}")
    else:
        ta = true_auc if true_auc is not None else float('nan')
        pa = pred_auc if pred_auc is not None else float('nan')
        row = (f"{epoch:>5}  {g(train_metric,7)}  {g(train_loss,10)}  "
               f"{g(ta,5)}  {g(pa,6)}  {g(val_metric,5)}  "
               f"{g(val_loss,8)}  {g(gradnorms,9)}  {g(elapsed,8,1)}")
    return row + ("  ✓" if saved else "")


class VNNTrainer():

	def __init__(self, data_wrapper):
		self.data_wrapper = data_wrapper
		self.train_feature = self.data_wrapper.train_feature
		self.train_label = self.data_wrapper.train_label
		self.val_feature = self.data_wrapper.val_feature
		self.val_label = self.data_wrapper.val_label
		self.task = self.data_wrapper.task


	def _get_loss_fn(self):
		if self.task == 'binary':
			return nn.BCEWithLogitsLoss()
		else:
			return nn.MSELoss()


	def _get_aux_loss_fn(self):
		"""Loss for auxiliary term outputs. Always use MSE — CCC is unstable
		with near-constant predictions from auxiliary heads early in training."""
		if self.task == 'binary':
			return nn.BCEWithLogitsLoss()
		else:
			return nn.MSELoss()


	def _compute_metrics(self, predictions, labels):
		"""Return (primary_metric_value, metric_name) for logging."""
		if self.task == 'binary':
			probs = torch.sigmoid(predictions)
			preds_binary = (probs >= 0.5).float()
			correct = (preds_binary.view(-1) == labels.view(-1)).float()
			acc = correct.sum() / len(correct)
			return acc.item(), 'accuracy'
		else:
			corr = util.pearson_corr(predictions, labels)
			return corr, 'pearson_r'


	def train_model(self):

		mlflow_enabled = getattr(self.data_wrapper, 'mlflow_enabled', False)
		if mlflow_enabled:
			try:
				import mlflow as _mlflow
				from pathlib import Path
				_mlflow.set_experiment("nest_vnn")
				ctx = _mlflow.start_run()
				# Persist run_id so predict step can link back to this run
				run_id_path = Path(self.data_wrapper.modeldir) / "mlflow_run_id.txt"
				run_id_path.write_text(ctx.info.run_id)
			except ImportError:
				print("Warning: mlflow not installed; disabling MLflow logging.")
				mlflow_enabled = False
				ctx = nullcontext()
		else:
			ctx = nullcontext()

		with ctx:
			return self._train_model_inner(mlflow_enabled)


	def _train_model_inner(self, mlflow_enabled):
		if mlflow_enabled:
			import mlflow, json
			from pathlib import Path
			from mlflow.data.http_dataset_source import HTTPDatasetSource
			from mlflow.data.meta_dataset import MetaDataset
			dw = self.data_wrapper
			# Derive study_id and read metadata written by cbioport_transform.py
			train_path = Path(dw.train)
			ndir = train_path.parent          # .../nest_vnn_input/
			study_id = ndir.parent.name       # .../output/<study_id>/nest_vnn_input
			meta_path = ndir / "metadata.json"
			meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

			params = {
				"study_id":            meta.get("study_id", study_id),
				"label_col":           dw.label_col,
				"task":                dw.task,
				"lr":                  dw.lr,
				"wd":                  dw.wd,
				"alpha":               dw.alpha,
				"epochs":              dw.epochs,
				"batchsize":           dw.batchsize,
				"num_hiddens_genotype": dw.num_hiddens_genotype,
				"dropout_fraction":    dw.dropout_fraction,
				"min_dropout_layer":   dw.min_dropout_layer,
				"zscore_method":       dw.zscore_method,
				"patience":            dw.patience,
				"delta":               dw.delta,
				"num_terms":           len(dw.dG.nodes()),
				"num_genes":           len(dw.gene_id_mapping),
				"num_train_samples":   len(self.train_feature),
				"num_val_samples":     len(self.val_feature),
			}
			if meta.get("ndex_uuid"):
				params["ndex_uuid"] = meta["ndex_uuid"]
			if meta.get("min_alt_freq") is not None:
				params["min_alt_freq"] = meta["min_alt_freq"]
			if meta.get("gene_count"):
				params["gene_count"] = meta["gene_count"]
			mlflow.log_params(params)
			mlflow.set_tags({
				"study_id": meta.get("study_id", study_id),
				"task":     dw.task,
				"label":    dw.label_col,
			})

			# Log cBioPortal study and NDEx hierarchy as dataset inputs
			if meta.get("cbioportal_url"):
				src = HTTPDatasetSource(url=meta["cbioportal_url"])
				mlflow.log_input(MetaDataset(source=src, name=study_id), context="cbioportal_study")
			if meta.get("ndex_url"):
				src = HTTPDatasetSource(url=meta["ndex_url"])
				mlflow.log_input(MetaDataset(source=src, name=meta.get("ndex_uuid", "ontology")), context="ontology")

		self.model = DrugCellNN(self.data_wrapper)
		self.model.to(self.data_wrapper.device)

		if mlflow_enabled:
			import mlflow
			from pathlib import Path
			dw = self.data_wrapper
			arch_png = Path(dw.modeldir) / "nn_architecture.png"
			try:
				_render_nn_architecture(
					dw.dG, dw.term_size_map, dw.term_direct_gene_map, dw.root, str(arch_png)
				)
				mlflow.log_artifact(str(arch_png), artifact_path=None)
			except Exception as e:
				print(f"Warning: could not render NN architecture image: {e}")

			schematic_png = Path(dw.modeldir) / "nn_schematic.png"
			try:
				_render_nn_schematic(
					root=dw.root,
					num_terms=len(dw.dG.nodes()),
					num_genes=dw.cell_features.shape[1],
					num_features=dw.cell_features.shape[2],
					num_hiddens=dw.num_hiddens_genotype,
					task=dw.task,
					out_path=str(schematic_png),
				)
				mlflow.log_artifact(str(schematic_png), artifact_path=None)
			except Exception as e:
				print(f"Warning: could not render NN schematic: {e}")

		epoch_start_time = time.time()
		min_loss = None
		best_val_metric = None
		best_train_metric = None
		best_val_loss = None
		best_train_loss = None
		best_epoch = None

		early_stopping_counter = 0
		term_mask_map = util.create_term_mask(self.model.term_direct_gene_map, self.model.gene_dim, self.data_wrapper.device)
		for name, param in self.model.named_parameters():
			if '_direct_gene_layer.weight' in name:
				term_name = name.split('_direct_gene_layer')[0]
				param.data = torch.mul(param.data, term_mask_map[term_name]) * 0.1
			else:
				param.data = param.data * 0.1

		train_loader = du.DataLoader(du.TensorDataset(self.train_feature, self.train_label), batch_size=self.data_wrapper.batchsize, shuffle=True, drop_last=False)
		val_loader = du.DataLoader(du.TensorDataset(self.val_feature, self.val_label), batch_size=self.data_wrapper.batchsize, shuffle=False)

		optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.data_wrapper.lr, betas=(0.9, 0.99), eps=1e-05, weight_decay=self.data_wrapper.wd)
		optimizer.zero_grad()

		for epoch in range(self.data_wrapper.epochs):
			# Train
			self.model.train()
			train_predict = torch.zeros(0, 0).to(self.data_wrapper.device)
			_gradnorms = torch.zeros(len(train_loader)).to(self.data_wrapper.device)
			epoch_train_loss = 0.0
			n_train_batches = 0

			for i, (inputdata, labels) in enumerate(train_loader):
				features = util.build_input_vector(inputdata, self.data_wrapper.cell_features)
				cuda_features = Variable(features.to(self.data_wrapper.device))
				cuda_labels = Variable(labels.to(self.data_wrapper.device))
				clinical_batch = util.build_clinical_vector(inputdata, self.data_wrapper.clinical_features).to(self.data_wrapper.device) if self.data_wrapper.num_clinical_features > 0 else None

				optimizer.zero_grad()

				aux_out_map, _ = self.model(cuda_features, clinical=clinical_batch)

				if train_predict.size()[0] == 0:
					train_predict = aux_out_map['final'].data
					train_label_gpu = cuda_labels
				else:
					train_predict = torch.cat([train_predict, aux_out_map['final'].data], dim=0)
					train_label_gpu = torch.cat([train_label_gpu, cuda_labels], dim=0)

				total_loss = 0
				loss_fn = self._get_loss_fn()
				aux_loss_fn = self._get_aux_loss_fn()
				for name, output in aux_out_map.items():
					if name == 'final':
						total_loss += loss_fn(output, cuda_labels)
					else:
						aux_loss = aux_loss_fn(output, cuda_labels)
						if not torch.isnan(aux_loss):
							total_loss += self.data_wrapper.alpha * aux_loss

				if torch.is_tensor(total_loss) and not torch.isnan(total_loss):
					total_loss.backward()
					epoch_train_loss += total_loss.item()
					n_train_batches += 1
				else:
					# Skip this batch if loss is NaN
					continue

				for name, param in self.model.named_parameters():
					if '_direct_gene_layer.weight' not in name:
						continue
					term_name = name.split('_direct_gene_layer')[0]
					param.grad.data = torch.mul(param.grad.data, term_mask_map[term_name])

				# Clip gradients to prevent NaN propagation
				torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
				_gradnorms[i] = util.get_grad_norm(self.model.parameters(), 2.0).unsqueeze(0)
				optimizer.step()

			gradnorms = sum(_gradnorms).unsqueeze(0).cpu().numpy()[0]
			epoch_train_loss = epoch_train_loss / max(n_train_batches, 1)
			if train_predict.size()[0] == 0:
				train_metric = float('nan')
			else:
				train_metric, _ = self._compute_metrics(train_predict, train_label_gpu)

			self.model.eval()

			val_predict = torch.zeros(0, 0).to(self.data_wrapper.device)
			epoch_val_loss = 0.0
			n_val_batches = 0

			with torch.no_grad():
				for i, (inputdata, labels) in enumerate(val_loader):
					features = util.build_input_vector(inputdata, self.data_wrapper.cell_features)
					cuda_features = Variable(features.to(self.data_wrapper.device))
					cuda_labels = Variable(labels.to(self.data_wrapper.device))
					clinical_batch = util.build_clinical_vector(inputdata, self.data_wrapper.clinical_features).to(self.data_wrapper.device) if self.data_wrapper.num_clinical_features > 0 else None

					aux_out_map, _ = self.model(cuda_features, clinical=clinical_batch)

					if val_predict.size()[0] == 0:
						val_predict = aux_out_map['final'].data
						val_label_gpu = cuda_labels
					else:
						val_predict = torch.cat([val_predict, aux_out_map['final'].data], dim=0)
						val_label_gpu = torch.cat([val_label_gpu, cuda_labels], dim=0)

					loss_fn = self._get_loss_fn()
					for name, output in aux_out_map.items():
						if name == 'final':
							epoch_val_loss += loss_fn(output, cuda_labels).item()
							n_val_batches += 1

			epoch_val_loss = epoch_val_loss / max(n_val_batches, 1)
			val_metric, metric_name = self._compute_metrics(val_predict, val_label_gpu)

			epoch_end_time = time.time()

			if self.task != 'binary':
				true_auc = float(torch.mean(train_label_gpu)) if train_predict.size()[0] > 0 else float('nan')
				pred_auc = float(torch.mean(train_predict)) if train_predict.size()[0] > 0 else float('nan')
			else:
				true_auc = pred_auc = None

			elapsed = epoch_end_time - epoch_start_time
			epoch_start_time = epoch_end_time

			saved = (min_loss is None or epoch_val_loss < min_loss - self.data_wrapper.delta)

			if epoch % _HEADER_REPEAT == 0:
				if epoch > 0:
					print()
				_print_epoch_header(self.task)
			print(_fmt_epoch_row(
				self.task, epoch, train_metric, epoch_train_loss,
				val_metric, epoch_val_loss, gradnorms, elapsed,
				saved=saved, true_auc=true_auc, pred_auc=pred_auc,
			))

			if saved:
				min_loss = epoch_val_loss
				best_val_metric = val_metric
				best_train_metric = train_metric
				best_val_loss = epoch_val_loss
				best_train_loss = epoch_train_loss
				best_epoch = epoch
				early_stopping_counter = 0
				torch.save(self.model, self.data_wrapper.modeldir + '/model_final.pt')
			else:
				early_stopping_counter += 1
				if early_stopping_counter >= self.data_wrapper.patience:
					print(f"\nEarly stopping at epoch {epoch} "
					      f"(no improvement for {self.data_wrapper.patience} epochs)")
					break

			if mlflow_enabled:
				import mlflow
				metrics = {
					f"train_{metric_name}": train_metric,
					"train_loss": epoch_train_loss,
					f"val_{metric_name}": val_metric,
					"val_loss": epoch_val_loss,
					"grad_norm": float(gradnorms),
				}
				if best_val_metric is not None:
					metrics[f"best_val_{metric_name}"] = best_val_metric
					metrics[f"best_train_{metric_name}"] = best_train_metric
					metrics["best_val_loss"] = best_val_loss
					metrics["best_train_loss"] = best_train_loss
					metrics["best_epoch"] = best_epoch
				mlflow.log_metrics(metrics, step=epoch)

		if mlflow_enabled:
			import mlflow
			import matplotlib
			matplotlib.use('Agg')
			import matplotlib.pyplot as plt
			from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

			# Load the best checkpoint once — used for confusion matrices and model logging.
			best_model = None
			if min_loss is not None:
				best_model = torch.load(
					self.data_wrapper.modeldir + '/model_final.pt',
					map_location=self.data_wrapper.device,
					weights_only=False,
				)
				best_model.to(self.data_wrapper.device)
				best_model.eval()

			if self.task == 'binary' and best_model is not None:
				for split_name, features, labels in [
					('train', self.train_feature, self.train_label),
					('val',   self.val_feature,   self.val_label),
				]:
					all_preds, all_labels = [], []
					loader = du.DataLoader(
						du.TensorDataset(features, labels),
						batch_size=self.data_wrapper.batchsize,
						shuffle=False,
					)
					with torch.no_grad():
						for inputdata, batch_labels in loader:
							feats = util.build_input_vector(inputdata, self.data_wrapper.cell_features)
							clin = util.build_clinical_vector(inputdata, self.data_wrapper.clinical_features).to(self.data_wrapper.device) if self.data_wrapper.num_clinical_features > 0 else None
							aux_out_map, _ = best_model(feats.to(self.data_wrapper.device), clinical=clin)
							probs = torch.sigmoid(aux_out_map['final'])
							preds = (probs >= 0.5).float().cpu().numpy().flatten()
							all_preds.extend(preds.tolist())
							all_labels.extend(batch_labels.numpy().flatten().tolist())

					cm = confusion_matrix(all_labels, all_preds)
					tn, fp, fn, tp = cm.ravel()
					mlflow.log_metrics({
						f"{split_name}_cm_tn": int(tn),
						f"{split_name}_cm_fp": int(fp),
						f"{split_name}_cm_fn": int(fn),
						f"{split_name}_cm_tp": int(tp),
					})

					disp = ConfusionMatrixDisplay(confusion_matrix=cm)
					fig, ax = plt.subplots(figsize=(4, 4))
					disp.plot(ax=ax, colorbar=False)
					ax.set_title(f'Best model — {split_name} split')
					fig.tight_layout()
					mlflow.log_figure(fig, f'confusion_matrix_{split_name}.png')
					plt.close(fig)

			# Log model in MLflow model format (not just a raw artifact).
			# This enables: model registry, versioning, and mlflow.pytorch.load_model().
			if best_model is not None:
				try:
					import mlflow.pytorch
					import mlflow.models

					# Infer input/output signature from a small val-set forward pass.
					# Input: the processed feature tensor produced by build_input_vector
					#        (shape: n_samples × n_genes × n_features, flattened for the model).
					# Output: raw scalar logits (binary) or z-scores (continuous) per sample.
					n_sig = min(8, len(self.val_feature))
					sig_input = util.build_input_vector(
						self.val_feature[:n_sig], self.data_wrapper.cell_features
					)
					sig_clin = util.build_clinical_vector(self.val_feature[:n_sig], self.data_wrapper.clinical_features).to(self.data_wrapper.device) if self.data_wrapper.num_clinical_features > 0 else None
					with torch.no_grad():
						sig_aux, _ = best_model(sig_input.to(self.data_wrapper.device), clinical=sig_clin)
					sig_out = sig_aux['final'].cpu().numpy()
					signature = mlflow.models.infer_signature(sig_input.numpy(), sig_out)

					mlflow.pytorch.log_model(best_model.cpu(), name="model",
					                         signature=signature, step=best_epoch)
				except Exception as e:
					print(f"Warning: mlflow.pytorch.log_model failed ({e}); "
					      f"falling back to log_artifact.")
					mlflow.log_artifact(self.data_wrapper.modeldir + '/model_final.pt',
					                    name="model")

			mlflow.log_artifact(self.data_wrapper.std)
			mlflow.log_artifact(self.data_wrapper.gene2id, artifact_path="input")
			mlflow.log_artifact(self.data_wrapper.onto, artifact_path="input")
			for subdir, readme in [
				("input/nest_vnn_input",    Path(self.data_wrapper.onto).parent / "README.md"),
				("input/cbioportal_output", Path(self.data_wrapper.onto).parent.parent / "cbioportal_output" / "README.md"),
			]:
				if readme.exists():
					mlflow.log_artifact(str(readme), artifact_path=subdir)

		return min_loss