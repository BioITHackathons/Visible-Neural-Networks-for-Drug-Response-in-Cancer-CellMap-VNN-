"""
NeST-VNN Explainability: RLIPP Scores + Annotated Hierarchy Visualization

Computes system-level importance (RLIPP) scores from prediction hidden embeddings,
and generates an annotated interactive hierarchy visualization.

This is a standalone version inspired by CM4AI's annotate.py, without CX2/NDEx
dependencies. It produces:
  1. rlipp_scores.txt     — RLIPP scores per ontology term
  2. gene_scores.txt      — Gene-level correlation scores
  3. hierarchy_annotated.graphml — NetworkX graph with RLIPP as node attributes
  4. hierarchy_viz.html    — Interactive HTML visualization

Usage:
    # After running predict.py with hidden output:
    python annotate_hierarchy.py \\
        -hidden result/hidden/ \\
        -ontology nest_vnn_input/ontology.txt \\
        -test nest_vnn_input/training_data.txt \\
        -predicted result/predict.txt \\
        -gene2id nest_vnn_input/gene2ind.txt \\
        -cell2id nest_vnn_input/cell2ind.txt \\
        -label binary_os -task binary \\
        -outdir explainability/ \\
        -genotype_hiddens 4 \\
        -cpu_count 4
"""

import argparse
import os
import numpy as np
import pandas as pd
import time
import json
import warnings
from pathlib import Path
from scipy import stats
from multiprocessing import Pool
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV

warnings.filterwarnings('ignore')

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

try:
    from ndex2.cx2 import CX2Network
    HAS_NDEX2 = True
except ImportError:
    HAS_NDEX2 = False


# ── RLIPP Calculator (adapted for clinical prediction) ──────────────────────

class ClinicalRLIPPCalculator:
    """
    Compute RLIPP (Relative Local Improvement in Predictive Power) scores.

    Adapted from NeST-VNN's rlipp_calculator.py for clinical outcome prediction
    (no drug grouping — all samples treated as one group).
    """

    def __init__(self, args):
        self.ontology = pd.read_csv(
            args.ontology, sep='\t', header=None,
            names=['S', 'T', 'I'], dtype={0: str, 1: str, 2: str}
        )
        self.terms = self.ontology['S'].unique().tolist()
        self.genes = pd.read_csv(
            args.gene2id, sep='\t', header=None, names=['I', 'G']
        )['G']
        self.cell_index = pd.read_csv(
            args.cell2id, sep='\t', header=None, names=['I', 'C']
        )

        # Load predicted values
        self.predicted_vals = np.loadtxt(args.predicted)

        # Load test labels
        self._load_test_labels(args)

        self.hidden_dir = args.hidden.rstrip('/') + '/'
        self.num_hiddens_genotype = args.genotype_hiddens or self._detect_hiddens(self.hidden_dir, self.terms)
        self.cpu_count = args.cpu_count
        self.outdir = Path(args.outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _detect_hiddens(hidden_dir: str, terms: list, fallback: int = 4) -> int:
        """Infer genotype_hiddens from the column count of the first readable term hidden file."""
        for term in terms:
            path = hidden_dir + term + '.hidden'
            if os.path.exists(path):
                try:
                    row = np.loadtxt(path, max_rows=1)
                    return int(row.size)
                except Exception:
                    continue
        return fallback

    def _load_test_labels(self, args):
        """Load test labels, handling both legacy and new header formats."""
        with open(args.test) as f:
            first_line = f.readline().strip()

        if 'cell_line' in first_line:
            # New header format
            df = pd.read_csv(args.test, sep='\t')
            label_col = args.label if args.label else 'binary_os'
            df = df.dropna(subset=[label_col])
            self.test_labels = df[label_col].astype(float).values
        else:
            # Legacy 4-column format
            df = pd.read_csv(args.test, sep='\t', header=None,
                             names=['C', 'D', 'AUC', 'DS'])
            self.test_labels = df['AUC'].values

        print(f"Loaded {len(self.test_labels)} test labels, "
              f"{len(self.predicted_vals)} predictions")

    def load_feature(self, element, size):
        file_name = self.hidden_dir + element + '.hidden'
        if not os.path.exists(file_name):
            return None
        return np.loadtxt(file_name, usecols=range(size))

    def load_all_features(self):
        """Load hidden embeddings for all terms and genes."""
        feature_map = {}

        print("Loading term features ...")
        with Pool(self.cpu_count) as p:
            results = p.starmap(
                self.load_feature,
                [(t, self.num_hiddens_genotype) for t in self.terms]
            )
        for i, t in enumerate(self.terms):
            if results[i] is not None:
                feature_map[t] = results[i]

        print("Loading gene features ...")
        with Pool(self.cpu_count) as p:
            results = p.starmap(
                self.load_feature, [(g, 1) for g in self.genes]
            )
        for i, g in enumerate(self.genes):
            if results[i] is not None:
                feature_map[g] = results[i]

        # Build child feature map
        child_feature_map = {}
        for term in self.terms:
            children = [
                row['T'] for _, row in self.ontology.iterrows()
                if row['S'] == term
            ]
            child_feature_map[term] = [
                feature_map[c] for c in children if c in feature_map
            ]

        print(f"Loaded features for {len([t for t in self.terms if t in feature_map])}/{len(self.terms)} terms, "
              f"{len([g for g in self.genes if g in feature_map])}/{len(self.genes)} genes")
        return feature_map, child_feature_map

    def exec_lm(self, X, y):
        """Ridge regression with PCA → Spearman correlation."""
        if X.shape[0] < 5 or X.shape[1] == 0:
            return 0.0, 1.0
        n_components = min(self.num_hiddens_genotype, X.shape[0], X.shape[1])
        if n_components < 1:
            return 0.0, 1.0
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X)
        regr = RidgeCV(cv=min(5, X.shape[0]))
        regr.fit(X_pca, y)
        y_pred = regr.predict(X_pca)
        return stats.spearmanr(y_pred, y)

    def calc_term_rlipp(self, term_features, term_child_features, term):
        """Calculate RLIPP for a single term (no drug grouping)."""
        X_parent = term_features
        if len(term_child_features) == 0:
            return None
        X_child = np.column_stack(term_child_features)
        y = self.predicted_vals

        # Trim to matching length
        n = min(X_parent.shape[0], X_child.shape[0], len(y))
        X_parent = X_parent[:n]
        X_child = X_child[:n]
        y = y[:n]

        p_rho, p_pval = self.exec_lm(X_parent, y)
        c_rho, c_pval = self.exec_lm(X_child, y)

        rlipp = p_rho / c_rho if abs(c_rho) > 1e-10 else 0.0

        return {
            'term': term,
            'p_rho': p_rho,
            'p_pval': p_pval,
            'c_rho': c_rho,
            'c_pval': c_pval,
            'rlipp': rlipp,
        }

    def calc_gene_rho(self, gene_features, gene):
        """Correlation between gene embedding and predictions."""
        y = self.predicted_vals
        n = min(len(gene_features), len(y))
        rho, p_val = stats.spearmanr(gene_features[:n], y[:n])
        return {'gene': gene, 'rho': rho, 'p_val': p_val}

    def calc_subsystem_gene_weights(self, feature_map):
        """
        For each term, rank directly-annotated genes by how strongly their
        hidden embedding drives the term's first principal component.

        Follows DrugCell: find the most-varying direction of the term's state
        (PC1), then rank each directly-annotated gene by Spearman correlation
        with that direction. High |pc1_corr| means that gene's network
        activation is a primary driver of this system's state.
        """
        gene_rows = self.ontology[self.ontology['I'] == 'gene']
        ont_direct_genes = gene_rows.groupby('S')['T'].apply(list).to_dict()

        results = []
        for term in self.terms:
            if term not in feature_map:
                continue
            term_h = feature_map[term]          # (n_samples, n_hiddens)
            direct_genes = [g for g in ont_direct_genes.get(term, []) if g in feature_map]
            if not direct_genes or term_h.shape[0] < 5:
                continue

            try:
                pca = PCA(n_components=1)
                pc1 = pca.fit_transform(term_h).squeeze()   # (n_samples,)
            except Exception:
                continue

            n = len(pc1)
            for gene in direct_genes:
                g_vec = feature_map[gene].squeeze()[:n]
                if len(g_vec) < 5:
                    continue
                rho, p_val = stats.spearmanr(g_vec, pc1[:len(g_vec)])
                results.append({
                    'term': term,
                    'gene': gene,
                    'pc1_corr': round(float(rho), 4),
                    'pc1_corr_pval': round(float(p_val), 6),
                })

        if not results:
            return pd.DataFrame(columns=['term', 'gene', 'pc1_corr', 'pc1_corr_pval', 'rank'])

        df = pd.DataFrame(results)
        df = df.dropna(subset=['pc1_corr'])
        if df.empty:
            return pd.DataFrame(columns=['term', 'gene', 'pc1_corr', 'pc1_corr_pval', 'rank'])

        df['abs_corr'] = df['pc1_corr'].abs()
        df['rank'] = (
            df.groupby('term')['abs_corr']
            .rank(ascending=False, method='first')
            .astype(int)
        )
        return df.drop(columns=['abs_corr']).sort_values(['term', 'rank']).reset_index(drop=True)

    def calc_boolean_logic(self, feature_map):
        """
        Characterize parent-child subsystem relationships as Boolean logic gates.

        For each term with ≥2 direct term children, considers all pairs of
        children and tests whether the parent's hidden state can be approximated
        by a Boolean function of the two children's states (AND, OR, XOR, etc.).

        Follows Ma et al. 2018 (DCell/VNN): binarize each subsystem's state at
        the median of PC1, build a majority-vote truth table for each
        (child1, child2) combination, then match against 10 non-trivial Boolean
        functions. Excludes trios where any combination has <4 samples or >50%
        of all samples.
        """
        from itertools import combinations

        # Term-to-term children only (exclude gene-leaf edges)
        term_children = {}
        for _, row in self.ontology.iterrows():
            if row['I'] != 'gene':
                term_children.setdefault(row['S'], []).append(row['T'])

        # Precompute PC1 and binary (above/below median) states for all terms
        bin_map = {}
        for term in self.terms:
            if term not in feature_map:
                continue
            h = feature_map[term]
            if h.shape[0] < 8:
                continue
            try:
                pc1 = PCA(n_components=1).fit_transform(h).squeeze()
                bin_map[term] = (pc1 >= np.median(pc1)).astype(int)
            except Exception:
                pass

        # Non-trivial Boolean functions: tuple is (F(0,0), F(0,1), F(1,0), F(1,1))
        # Index encoding: child1_bin * 2 + child2_bin
        bool_functions = {
            'AND':        (0, 0, 0, 1),
            'OR':         (0, 1, 1, 1),
            'XOR':        (0, 1, 1, 0),
            'A_NOT_B':    (0, 0, 1, 0),   # child1 active, child2 inactive
            'B_NOT_A':    (0, 1, 0, 0),   # child2 active, child1 inactive
            'NOR':        (1, 0, 0, 0),
            'NAND':       (1, 1, 1, 0),
            'XNOR':       (1, 0, 0, 1),
            'A_OR_NOT_B': (1, 0, 1, 1),
            'B_OR_NOT_A': (1, 1, 0, 1),
        }
        tt_to_name = {v: k for k, v in bool_functions.items()}

        results = []
        for term in self.terms:
            if term not in bin_map:
                continue
            children = [c for c in term_children.get(term, []) if c in bin_map]
            if len(children) < 2:
                continue

            parent_bin = bin_map[term]
            n = len(parent_bin)

            for c1, c2 in combinations(children, 2):
                c1_bin = bin_map[c1][:n]
                c2_bin = bin_map[c2][:n]

                # votes[idx] = [count_parent_0, count_parent_1]
                votes = [[0, 0] for _ in range(4)]
                for i in range(n):
                    idx = int(c1_bin[i]) * 2 + int(c2_bin[i])
                    votes[idx][int(parent_bin[i])] += 1

                counts = [v[0] + v[1] for v in votes]

                # Exclusion criteria from DCell paper
                if any(c < 4 for c in counts) or any(c > n * 0.5 for c in counts):
                    continue

                # Majority-vote truth table
                tt = tuple(1 if v[1] >= v[0] else 0 for v in votes)
                fn_name = tt_to_name.get(tt)
                if fn_name is None:
                    continue

                correct = sum(
                    1 for i in range(n)
                    if parent_bin[i] == tt[int(c1_bin[i]) * 2 + int(c2_bin[i])]
                )
                results.append({
                    'parent': term,
                    'child1': c1,
                    'child2': c2,
                    'logic': fn_name,
                    'consistency': round(correct / n, 4),
                    'n_samples': n,
                })

        if not results:
            return pd.DataFrame(
                columns=['parent', 'child1', 'child2', 'logic', 'consistency', 'n_samples']
            )

        return (
            pd.DataFrame(results)
            .sort_values('consistency', ascending=False)
            .reset_index(drop=True)
        )

    def calc_scores(self):
        """Calculate all RLIPP and gene scores."""
        print("\nCalculating RLIPP scores ...")
        start = time.time()
        feature_map, child_feature_map = self.load_all_features()
        print(f"Features loaded in {time.time() - start:.1f}s")

        # Term RLIPP scores
        start = time.time()
        rlipp_results = []
        with Parallel(backend="multiprocessing", n_jobs=self.cpu_count) as parallel:
            results = parallel(
                delayed(self.calc_term_rlipp)(
                    feature_map[term], child_feature_map[term], term
                )
                for term in self.terms if term in feature_map
            )
            rlipp_results = [r for r in results if r is not None]

        rlipp_df = pd.DataFrame(rlipp_results)
        rlipp_df = rlipp_df.sort_values('rlipp', ascending=False)

        rlipp_path = self.outdir / 'rlipp_scores.txt'
        rlipp_df.to_csv(rlipp_path, sep='\t', index=False, encoding='utf-8')
        print(f"RLIPP scores: {len(rlipp_df)} terms → {rlipp_path}")

        # Gene correlation scores
        gene_results = []
        with Parallel(backend="multiprocessing", n_jobs=self.cpu_count) as parallel:
            results = parallel(
                delayed(self.calc_gene_rho)(feature_map[gene], gene)
                for gene in self.genes if gene in feature_map
            )
            gene_results = [r for r in results if r is not None]

        gene_df = pd.DataFrame(gene_results)
        gene_df = gene_df.sort_values('rho', ascending=False, key=abs)

        gene_path = self.outdir / 'gene_scores.txt'
        gene_df.to_csv(gene_path, sep='\t', index=False, encoding='utf-8')
        print(f"Gene scores:  {len(gene_df)} genes → {gene_path}")

        # Gene weights within subsystems (PC1 correlation)
        print("Calculating subsystem gene weights ...")
        subsys_gene_df = self.calc_subsystem_gene_weights(feature_map)

        subsys_gene_path = self.outdir / 'subsystem_gene_weights.txt'
        subsys_gene_df.to_csv(subsys_gene_path, sep='\t', index=False, float_format='%.4f', encoding='utf-8')
        print(f"Subsystem gene weights: {len(subsys_gene_df)} term-gene pairs → {subsys_gene_path}")

        # Top-systems summary: top 20 terms × top 5 driving genes
        if not subsys_gene_df.empty and not rlipp_df.empty:
            top_terms = rlipp_df.head(20)['term'].tolist()
            top_genes = (
                subsys_gene_df[
                    subsys_gene_df['term'].isin(top_terms) & (subsys_gene_df['rank'] <= 5)
                ]
                .merge(rlipp_df[['term', 'rlipp']], on='term', how='left')
                .sort_values(['rlipp', 'rank'], ascending=[False, True])
            )
            top_genes[['term', 'rlipp', 'gene', 'pc1_corr', 'pc1_corr_pval', 'rank']].to_csv(
                self.outdir / 'top_subsystem_genes.txt', sep='\t', index=False, float_format='%.4f'
            )
            print(f"Top subsystem genes → {self.outdir / 'top_subsystem_genes.txt'}")

        # Boolean logic characterization of subsystem trios
        print("Calculating Boolean logic characterization ...")
        bool_logic_df = self.calc_boolean_logic(feature_map)

        bool_logic_path = self.outdir / 'boolean_logic.txt'
        bool_logic_df.to_csv(bool_logic_path, sep='\t', index=False, float_format='%.4f', encoding='utf-8')
        print(f"Boolean logic: {len(bool_logic_df)} trios matched → {bool_logic_path}")

        if not bool_logic_df.empty:
            summary = bool_logic_df['logic'].value_counts()
            print("  Logic function counts: " + ", ".join(f"{k}={v}" for k, v in summary.items()))

        print(f"Scores computed in {time.time() - start:.1f}s")

        return rlipp_df, gene_df, subsys_gene_df, bool_logic_df


# ── Hierarchy Builder ────────────────────────────────────────────────────────

def build_annotated_hierarchy(ontology_path, rlipp_df, gene_df, outdir,
                              subsys_gene_df=None, bool_logic_df=None):
    """
    Build an annotated hierarchy graph from the ontology + RLIPP scores.
    Exports GraphML and interactive HTML.
    """
    ontology = pd.read_csv(
        ontology_path, sep='\t', header=None,
        names=['parent', 'child', 'relation'], dtype=str
    )

    # Build score lookups
    rlipp_scores = {}
    if not rlipp_df.empty:
        rlipp_scores = dict(zip(rlipp_df['term'], rlipp_df['rlipp']))

    gene_scores = {}
    if not gene_df.empty:
        gene_scores = dict(zip(gene_df['gene'], gene_df['rho']))

    # Identify terms vs genes
    parents = set(ontology['parent'])
    children = set(ontology['child'])
    genes = set(ontology[ontology['relation'] == 'gene']['child'])
    terms = (parents | children) - genes

    # Build graph
    if not HAS_NETWORKX:
        print("⚠ networkx not installed — skipping GraphML export")
        G = None
    else:
        G = nx.DiGraph()

        # Add term nodes
        for term in terms:
            attrs = {
                'node_type': 'term',
                'rlipp': float(rlipp_scores.get(term, 0.0)),
            }
            # Add p_rho, c_rho if available
            if not rlipp_df.empty and term in rlipp_scores:
                row = rlipp_df[rlipp_df['term'] == term].iloc[0]
                attrs['p_rho'] = float(row['p_rho'])
                attrs['c_rho'] = float(row['c_rho'])
                attrs['p_pval'] = float(row['p_pval'])
                attrs['c_pval'] = float(row['c_pval'])
            G.add_node(term, **attrs)

        # Add gene nodes
        for gene in genes:
            G.add_node(gene, **{
                'node_type': 'gene',
                'rho': float(gene_scores.get(gene, 0.0)),
            })

        # Add edges
        for _, row in ontology.iterrows():
            G.add_edge(row['parent'], row['child'], relation=row['relation'])

        # Export GraphML
        graphml_path = outdir / 'hierarchy_annotated.graphml'
        nx.write_graphml(G, graphml_path)
        print(f"GraphML:      {graphml_path} ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")

    # Export CX2
    build_cx2_hierarchy(ontology, terms, genes, rlipp_scores, rlipp_df, gene_scores, gene_df, outdir)

    # Build interactive HTML visualization
    html_path = outdir / 'hierarchy_viz.html'
    build_html_viz(ontology, terms, genes, rlipp_scores, rlipp_df, gene_scores, gene_df, html_path,
                   subsys_gene_df=subsys_gene_df, bool_logic_df=bool_logic_df)
    print(f"HTML viz:     {html_path}")

    # Summary table
    summary_path = outdir / 'top_systems.txt'
    if not rlipp_df.empty:
        top = rlipp_df.head(20)
        top.to_csv(summary_path, sep='\t', index=False, float_format='%.4f', encoding='utf-8')
        print(f"\nTop 20 systems by RLIPP:")
        for _, row in top.iterrows():
            print(f"  {row['term']:30s}  RLIPP={row['rlipp']:.3f}  "
                  f"P_rho={row['p_rho']:.3f}  C_rho={row['c_rho']:.3f}")

    return G


def build_cx2_hierarchy(ontology, terms, genes, rlipp_scores, rlipp_df, gene_scores, gene_df, outdir):
    """Build a CX2 hierarchy file compatible with Cytoscape Web and NDEx."""
    if not HAS_NDEX2:
        print("CX2:          ⚠ ndex2 not installed — skipping (pip install ndex2)")
        return

    import math

    def safe_float(val, default=0.0):
        """Convert to float, replacing NaN/inf with default."""
        try:
            f = float(val)
            if math.isnan(f) or math.isinf(f):
                return default
            return f
        except (ValueError, TypeError):
            return default

    # Build gene p_val lookup
    gene_pvals = {}
    if not gene_df.empty and 'p_val' in gene_df.columns:
        gene_pvals = dict(zip(gene_df['gene'], gene_df['p_val']))

    # Build parent→children map from ontology for recursive gene collection
    ont_children = {}
    ont_gene_children = {}
    for _, row in ontology.iterrows():
        parent = row['parent']
        child = row['child']
        relation = row['relation']
        if relation == 'gene':
            ont_gene_children.setdefault(parent, []).append(child)
        else:
            ont_children.setdefault(parent, []).append(child)

    def get_all_descendant_genes(term, visited=None):
        if visited is None:
            visited = set()
        if term in visited:
            return []
        visited.add(term)
        result = list(ont_gene_children.get(term, []))
        for child_term in ont_children.get(term, []):
            result.extend(get_all_descendant_genes(child_term, visited))
        return result

    net = CX2Network()
    net.set_network_attributes({
        'name': 'NeST-VNN Annotated Hierarchy',
        'description': 'Ontology hierarchy annotated with RLIPP system importance scores',
    })

    # Track node name → CX2 node ID mapping
    name_to_id = {}

    # Add term nodes
    for term in sorted(terms):
        node_id = net.add_node(attributes={'name': term, 'type': 'term'})
        name_to_id[term] = node_id

        rlipp = safe_float(rlipp_scores.get(term, 0.0))
        net.add_node_attribute(node_id, 'RLIPP', rlipp, datatype='double')

        if not rlipp_df.empty and term in rlipp_scores:
            row = rlipp_df[rlipp_df['term'] == term]
            if not row.empty:
                row = row.iloc[0]
                net.add_node_attribute(node_id, 'P_rho', safe_float(row['p_rho']), datatype='double')
                net.add_node_attribute(node_id, 'P_pval', safe_float(row['p_pval'], 1.0), datatype='double')
                net.add_node_attribute(node_id, 'C_rho', safe_float(row['c_rho']), datatype='double')
                net.add_node_attribute(node_id, 'C_pval', safe_float(row['c_pval'], 1.0), datatype='double')

        # All descendant genes (recursive)
        all_desc_genes = sorted(set(get_all_descendant_genes(term)))
        net.add_node_attribute(node_id, 'gene_count', len(all_desc_genes), datatype='integer')
        if all_desc_genes:
            net.add_node_attribute(node_id, 'descendant_genes',
                                   all_desc_genes, datatype='list_of_string')

    # Add gene nodes
    for gene in sorted(genes):
        node_id = net.add_node(attributes={'name': gene, 'type': 'gene'})
        name_to_id[gene] = node_id

        rho = safe_float(gene_scores.get(gene, 0.0))
        net.add_node_attribute(node_id, 'rho', rho, datatype='double')

        p_val = safe_float(gene_pvals.get(gene, 1.0), 1.0)
        net.add_node_attribute(node_id, 'p_val', p_val, datatype='double')

    # Add edges
    for _, row in ontology.iterrows():
        parent = row['parent']
        child = row['child']
        if parent in name_to_id and child in name_to_id:
            net.add_edge(source=name_to_id[parent], target=name_to_id[child],
                         attributes={'interaction': row['relation']})

    # Visual style: terms colored by RLIPP (grey→yellow→orange), genes as small diamonds,
    # gene edges dashed. RLIPP gradient is anchored at 1.0 (neutral) so terms that add
    # information above their children stand out in warm colors.
    max_rlipp = max((v for v in rlipp_scores.values() if v == v), default=2.0)
    max_rlipp = max(max_rlipp, 2.0)
    net.set_visual_properties({
        "default": {
            "network": {"NETWORK_BACKGROUND_COLOR": "#FFFFFF"},
            "node": {
                "NODE_FILL_COLOR": "#CCCCCC",
                "NODE_SHAPE": "ELLIPSE",
                "NODE_SIZE": 50.0,
                "NODE_LABEL_FONT_SIZE": 10,
                "NODE_BORDER_COLOR": "#555555",
                "NODE_BORDER_WIDTH": 1.5,
            },
            "edge": {
                "EDGE_LINE_TYPE": "SOLID",
                "EDGE_WIDTH": 1.0,
                "EDGE_STROKE_UNSELECTED_PAINT": "#888888",
            },
        },
        "nodeMapping": {
            "NODE_LABEL": {
                "type": "PASSTHROUGH",
                "definition": {"attribute": "name"},
            },
            "NODE_SHAPE": {
                "type": "DISCRETE",
                "definition": {
                    "attribute": "type",
                    "map": [
                        {"v": "term", "vp": "ELLIPSE"},
                        {"v": "gene", "vp": "DIAMOND"},
                    ],
                },
            },
            "NODE_SIZE": {
                "type": "DISCRETE",
                "definition": {
                    "attribute": "type",
                    "map": [
                        {"v": "term", "vp": 50.0},
                        {"v": "gene", "vp": 18.0},
                    ],
                },
            },
            # Gradient: RLIPP=0 (gene nodes) → grey; 1.0 (neutral) → light yellow; max → dark orange
            "NODE_FILL_COLOR": {
                "type": "CONTINUOUS",
                "definition": {
                    "attribute": "RLIPP",
                    "map": [
                        {"includeMin": True, "includeMax": False,
                         "min": 0.0, "max": 1.0,
                         "minVP": "#CCCCCC", "maxVP": "#FFFFD4"},
                        {"includeMin": True, "includeMax": True,
                         "min": 1.0, "max": max_rlipp,
                         "minVP": "#FFFFD4", "maxVP": "#D94801"},
                    ],
                },
            },
            "NODE_BORDER_WIDTH": {
                "type": "DISCRETE",
                "definition": {
                    "attribute": "type",
                    "map": [
                        {"v": "term", "vp": 1.5},
                        {"v": "gene", "vp": 0.5},
                    ],
                },
            },
        },
        "edgeMapping": {
            "EDGE_LINE_TYPE": {
                "type": "DISCRETE",
                "definition": {
                    "attribute": "interaction",
                    "map": [
                        {"v": "default", "vp": "SOLID"},
                        {"v": "gene",    "vp": "LONG_DASH"},
                    ],
                },
            },
            "EDGE_STROKE_UNSELECTED_PAINT": {
                "type": "DISCRETE",
                "definition": {
                    "attribute": "interaction",
                    "map": [
                        {"v": "default", "vp": "#555555"},
                        {"v": "gene",    "vp": "#AAAAAA"},
                    ],
                },
            },
        },
    })

    # Write CX2 file
    cx2_path = outdir / 'hierarchy_annotated.cx2'
    net.write_as_raw_cx2(str(cx2_path))
    n_nodes = len(net.get_nodes())
    n_edges = len(net.get_edges())
    print(f"CX2:          {cx2_path} ({n_nodes} nodes, {n_edges} edges)")


def build_html_viz(ontology, terms, genes, rlipp_scores, rlipp_df, gene_scores, gene_df, outpath,
                   subsys_gene_df=None, bool_logic_df=None):
    """Build a standalone interactive HTML visualization of the annotated hierarchy."""
    import math

    def sf(val, default=0.0):
        try:
            f = float(val)
            return default if math.isnan(f) or math.isinf(f) else round(f, 4)
        except (ValueError, TypeError):
            return default

    # Build RLIPP lookup by term
    rlipp_rows = {}
    if not rlipp_df.empty:
        for _, row in rlipp_df.iterrows():
            rlipp_rows[row['term']] = row

    # Build gene p_val lookup
    gene_pvals = {}
    if not gene_df.empty and 'p_val' in gene_df.columns:
        gene_pvals = dict(zip(gene_df['gene'], gene_df['p_val']))

    # Prepare data for JavaScript
    nodes = []
    for term in sorted(terms):
        node = {
            'id': term, 'type': 'term', 'label': term,
            'rlipp': sf(rlipp_scores.get(term, 0.0)),
            'p_rho': 0.0, 'p_pval': 1.0,
            'c_rho': 0.0, 'c_pval': 1.0,
        }
        if term in rlipp_rows:
            r = rlipp_rows[term]
            node['p_rho'] = sf(r.get('p_rho', 0.0))
            node['p_pval'] = sf(r.get('p_pval', 1.0), 1.0)
            node['c_rho'] = sf(r.get('c_rho', 0.0))
            node['c_pval'] = sf(r.get('c_pval', 1.0), 1.0)
        nodes.append(node)

    # Build ontology maps for pre-computing gene lists per term.
    # Note: the NeST ontology propagates gene annotations upward, so every ancestor
    # term directly lists ALL genes from its subtree. We therefore cannot use
    # "direct vs inherited" naively — instead we split into:
    #   term-specific genes: in this term's list but NOT in any immediate child term
    #   child-system genes:  in any immediate child term's list
    ont_children = {}
    ont_gene_children = {}
    for _, row in ontology.iterrows():
        parent, child, relation = row['parent'], row['child'], row['relation']
        if relation == 'gene':
            ont_gene_children.setdefault(parent, []).append(child)
        else:
            ont_children.setdefault(parent, []).append(child)

    for node in nodes:
        term = node['id']
        own_genes = set(ont_gene_children.get(term, []))
        child_genes: set = set()
        for child_term in ont_children.get(term, []):
            child_genes.update(ont_gene_children.get(child_term, []))

        specific = sorted(own_genes - child_genes)
        node['direct_genes'] = [
            {'id': g, 'rho': sf(gene_scores.get(g, 0.0)), 'p_val': sf(gene_pvals.get(g, 1.0), 1.0)}
            for g in specific
        ]
        inherited = sorted(
            child_genes,
            key=lambda g: abs(gene_scores.get(g, 0.0)), reverse=True
        )
        node['descendant_genes'] = [
            {'id': g, 'rho': sf(gene_scores.get(g, 0.0)), 'p_val': sf(gene_pvals.get(g, 1.0), 1.0)}
            for g in inherited
        ]

    # Build driving_genes lookup: term -> top-10 genes by |pc1_corr|
    driving_genes_map = {}
    if subsys_gene_df is not None and not subsys_gene_df.empty:
        gene_pvals_lookup = dict(zip(gene_df['gene'], gene_df['p_val'])) if 'p_val' in gene_df.columns else {}
        for term, grp in subsys_gene_df.groupby('term'):
            top = grp[grp['rank'] <= 10].copy()
            driving_genes_map[term] = [
                {
                    'id': row['gene'],
                    'pc1_corr': sf(row['pc1_corr']),
                    'pc1_corr_pval': sf(row['pc1_corr_pval'], 1.0),
                    'rho': sf(gene_scores.get(row['gene'], 0.0)),
                    'rho_pval': sf(gene_pvals_lookup.get(row['gene'], 1.0), 1.0),
                }
                for _, row in top.iterrows()
            ]

    for node in nodes:
        node['driving_genes'] = driving_genes_map.get(node['id'], [])

    # Build boolean logic lookup: parent_term -> [{child1, child2, logic, consistency}]
    bool_logic_map = {}
    if bool_logic_df is not None and not bool_logic_df.empty:
        for _, row in bool_logic_df.iterrows():
            entry = {
                'child1': row['child1'],
                'child2': row['child2'],
                'logic': row['logic'],
                'consistency': sf(row['consistency']),
                'n_samples': int(row['n_samples']),
            }
            bool_logic_map.setdefault(row['parent'], []).append(entry)

    for node in nodes:
        node['bool_logic'] = bool_logic_map.get(node['id'], [])

    # Only include term-to-term edges; gene data lives in node attributes above.
    edges = []
    for _, row in ontology.iterrows():
        if row['relation'] != 'gene':
            edges.append({
                'source': row['parent'], 'target': row['child'],
                'relation': row['relation']
            })

    nodes_json = json.dumps(nodes)
    edges_json = json.dumps(edges)

    # Compute RLIPP range for color scaling
    rlipp_vals = [s for s in rlipp_scores.values() if abs(s) > 0]
    max_rlipp = max(rlipp_vals) if rlipp_vals else 1.0
    total_genes = len(genes)

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>NeST-VNN Annotated Hierarchy</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       background: #0a0a0a; color: #e0e0e0; }}
#header {{ padding: 16px 24px; background: #1a1a1a; border-bottom: 1px solid #333;
           display: flex; justify-content: space-between; align-items: center; }}
#header h1 {{ font-size: 18px; font-weight: 600; }}
#header .stats {{ font-size: 13px; color: #888; }}
#controls {{ padding: 12px 24px; background: #141414; border-bottom: 1px solid #222;
             display: flex; gap: 16px; align-items: center; font-size: 13px; }}
#controls label {{ color: #aaa; }}
#controls input, #controls select {{ background: #222; border: 1px solid #444; color: #e0e0e0;
                                      padding: 4px 8px; border-radius: 4px; }}
#main {{ display: flex; height: calc(100vh - 100px); }}
#table-panel {{ width: 580px; overflow-y: auto; border-right: 1px solid #222;
                padding: 8px; font-size: 12px; }}
#table-panel table {{ width: 100%; border-collapse: collapse; }}
#table-panel th {{ text-align: left; padding: 6px 8px; border-bottom: 1px solid #333;
                   position: sticky; top: 0; background: #141414; color: #aaa; font-size: 11px; }}
#table-panel td {{ padding: 5px 8px; border-bottom: 1px solid #1a1a1a; cursor: pointer; }}
#table-panel tr:hover {{ background: #1a2a3a; }}
#table-panel tr.selected {{ background: #1a3a2a; }}
.bar {{ display: inline-block; height: 12px; border-radius: 2px; min-width: 2px; }}
#detail {{ flex: 1; padding: 24px; overflow-y: auto; }}
#detail h2 {{ font-size: 16px; margin-bottom: 12px; color: #7eb8da; }}
#detail .meta {{ color: #888; font-size: 13px; margin-bottom: 16px; }}
#detail .children {{ margin-top: 12px; }}
#detail .child {{ display: inline-block; background: #1a1a2e; padding: 4px 10px; margin: 3px;
                  border-radius: 4px; font-size: 12px; cursor: pointer; }}
#detail .child:hover {{ background: #2a2a4e; }}
#detail .child.gene {{ background: #1e2a1e; }}
.rlipp-high {{ color: #f5a623; }}
.rlipp-mid {{ color: #7eb8da; }}
.rlipp-low {{ color: #666; }}
</style>
</head>
<body>
<div id="header">
    <h1>NeST-VNN Annotated Hierarchy</h1>
    <div class="stats" id="stats"></div>
</div>
<div id="controls">
    <label>Sort by:</label>
    <select id="sort-select">
        <option value="rlipp">RLIPP (desc)</option>
        <option value="name">Name</option>
        <option value="p_rho">P_rho (desc)</option>
    </select>
    <label>Filter:</label>
    <input type="text" id="filter-input" placeholder="Search terms...">
    <label>Min RLIPP:</label>
    <input type="number" id="min-rlipp" value="0" step="0.1" style="width:70px">
</div>
<div id="main">
    <div id="table-panel"><table>
        <thead><tr><th>System</th><th>RLIPP</th><th>P_rho</th><th>P_pval</th><th>C_rho</th><th>C_pval</th><th></th></tr></thead>
        <tbody id="table-body"></tbody>
    </table></div>
    <div id="detail" id="detail-panel">
        <p style="color:#666">Select a system from the table to see details.</p>
    </div>
</div>
<script>
const nodes = {nodes_json};
const edges = {edges_json};
const maxRlipp = {max_rlipp};

const termNodes = nodes.filter(n => n.type === 'term');
const nodeMap = {{}};
nodes.forEach(n => nodeMap[n.id] = n);

// Build parent/child maps
const childrenOf = {{}};
const parentsOf = {{}};
edges.forEach(e => {{
    if (!childrenOf[e.source]) childrenOf[e.source] = [];
    childrenOf[e.source].push({{id: e.target, relation: e.relation}});
    if (!parentsOf[e.target]) parentsOf[e.target] = [];
    parentsOf[e.target].push(e.source);
}});

document.getElementById('stats').textContent =
    `${{termNodes.length}} systems · {total_genes} genes · ${{edges.length}} term edges`;

function rlippColor(val) {{
    if (val > 1.2) return '#f5a623';
    if (val > 1.0) return '#7eb8da';
    return '#555';
}}

function renderTable() {{
    const sort = document.getElementById('sort-select').value;
    const filter = document.getElementById('filter-input').value.toLowerCase();
    const minRlipp = parseFloat(document.getElementById('min-rlipp').value) || 0;

    let data = termNodes.filter(n => {{
        if (filter && !n.id.toLowerCase().includes(filter)) return false;
        if (n.rlipp < minRlipp) return false;
        return true;
    }});

    if (sort === 'rlipp') data.sort((a, b) => b.rlipp - a.rlipp);
    else if (sort === 'name') data.sort((a, b) => a.id.localeCompare(b.id));
    else if (sort === 'p_rho') data.sort((a, b) => (b.p_rho||0) - (a.p_rho||0));

    const tbody = document.getElementById('table-body');
    tbody.innerHTML = data.map(n => {{
        const w = Math.max(2, Math.min(120, (n.rlipp / maxRlipp) * 120));
        const c = rlippColor(n.rlipp);
        return `<tr onclick="showDetail('${{n.id}}')" data-id="${{n.id}}">
            <td>${{n.id}}</td>
            <td style="color:${{c}}">${{n.rlipp.toFixed(3)}}</td>
            <td>${{(n.p_rho||0).toFixed(3)}}</td>
            <td>${{(n.p_pval||0).toExponential(1)}}</td>
            <td>${{(n.c_rho||0).toFixed(3)}}</td>
            <td>${{(n.c_pval||0).toExponential(1)}}</td>
            <td><span class="bar" style="width:${{w}}px;background:${{c}}"></span></td>
        </tr>`;
    }}).join('');
}}

function showDetail(termId) {{
    document.querySelectorAll('#table-body tr').forEach(tr => {{
        tr.classList.toggle('selected', tr.dataset.id === termId);
    }});

    const node = nodeMap[termId];
    // All edges are term-to-term; gene lists are pre-computed in node attributes.
    const termChildren = childrenOf[termId] || [];
    const parents = parentsOf[termId] || [];
    const directGenes = node.direct_genes || [];
    const inheritedGenes = node.descendant_genes || [];

    let html = `<h2>${{termId}}</h2>`;
    html += `<div class="meta">`;
    html += `RLIPP: <strong style="color:${{rlippColor(node.rlipp)}}">${{node.rlipp.toFixed(4)}}</strong>`;
    if (node.p_rho !== undefined) html += ` &nbsp;|&nbsp; P_rho: ${{node.p_rho.toFixed(4)}}`;
    if (node.p_pval !== undefined) html += ` (p=${{node.p_pval.toExponential(2)}})`;
    if (node.c_rho !== undefined) html += ` &nbsp;|&nbsp; C_rho: ${{node.c_rho.toFixed(4)}}`;
    if (node.c_pval !== undefined) html += ` (p=${{node.c_pval.toExponential(2)}})`;
    html += `</div>`;

    if (parents.length > 0) {{
        html += `<div class="children"><strong>Parents (${{parents.length}}):</strong><br>`;
        parents.forEach(p => {{
            const pn = nodeMap[p];
            const rl = pn ? ` (RLIPP=${{pn.rlipp.toFixed(3)}})` : '';
            html += `<span class="child" onclick="showDetail('${{p}}')">${{p}}${{rl}}</span>`;
        }});
        html += `</div>`;
    }}

    if (termChildren.length > 0) {{
        html += `<div class="children"><strong>Child systems (${{termChildren.length}}):</strong><br>`;
        termChildren.forEach(c => {{
            const cn = nodeMap[c.id];
            const rl = cn ? ` (RLIPP=${{cn.rlipp.toFixed(3)}})` : '';
            html += `<span class="child" onclick="showDetail('${{c.id}}')">${{c.id}}${{rl}}</span>`;
        }});
        html += `</div>`;
    }}

    if (node.driving_genes && node.driving_genes.length > 0) {{
        html += `<div class="children" style="margin-top:14px"><strong>Key driving genes (${{node.driving_genes.length}}):</strong>`;
        html += ` <span style="color:#666;font-size:11px">ranked by |correlation with PC1 of system hidden embedding|</span><br>`;
        node.driving_genes.forEach((g, i) => {{
            const corrSign = g.pc1_corr >= 0 ? '+' : '';
            const rhoSign  = g.rho >= 0 ? '+' : '';
            const borderColor = g.pc1_corr >= 0 ? '#e08030' : '#50a870';
            const tip = `pc1_corr=${{corrSign}}${{g.pc1_corr.toFixed(3)}} (p=${{g.pc1_corr_pval.toExponential(1)}}), cohort ρ=${{rhoSign}}${{g.rho.toFixed(3)}}`;
            html += `<span class="child gene" style="border-left:3px solid ${{borderColor}}" title="${{tip}}">`;
            html += `#${{i+1}} ${{g.id}} <span style="color:#888;font-size:10px">${{corrSign}}${{g.pc1_corr.toFixed(3)}}</span></span>`;
        }});
        html += `</div>`;
    }}

    if (node.bool_logic && node.bool_logic.length > 0) {{
        const logicIcons = {{
            'AND':'A∧B', 'OR':'A∨B', 'XOR':'A⊕B', 'NOR':'¬(A∨B)', 'NAND':'¬(A∧B)',
            'XNOR':'A↔B', 'A_NOT_B':'A∧¬B', 'B_NOT_A':'B∧¬A',
            'A_OR_NOT_B':'A∨¬B', 'B_OR_NOT_A':'B∨¬A',
        }};
        const logicDesc = {{
            'AND':        'Co-requirement: both children must be active to activate parent',
            'OR':         'Redundancy: either child alone is sufficient to activate parent',
            'XOR':        'Mutual exclusivity: exactly one child active, not both simultaneously',
            'NOR':        'Dual inhibition: parent active only when both children are inactive',
            'NAND':       'Negative synergy: parent active unless both children are simultaneously on',
            'XNOR':       'Concordance: parent active when both children share the same state',
            'A_NOT_B':    'A dominates: parent active when A is on and B is off',
            'B_NOT_A':    'B dominates: parent active when B is on and A is off',
            'A_OR_NOT_B': 'A or not-B: parent active in all states except when B alone is on',
            'B_OR_NOT_A': 'B or not-A: parent active in all states except when A alone is on',
        }};
        html += `<div class="children" style="margin-top:14px">`;
        html += `<strong>Boolean logic relationships (${{node.bool_logic.length}})</strong> `;
        html += `<span style="color:#666;font-size:11px">parent state ≈ f(child1, child2) binarized at median PC1</span> `;
        html += `<span style="cursor:pointer;border:1px solid #334;border-radius:3px;padding:0 5px;font-size:10px;color:#668;margin-left:4px" onclick="var e=document.getElementById('bhelp-${{termId}}');e.style.display=e.style.display==='none'?'block':'none'">? help</span><br>`;
        html += `<div id="bhelp-${{termId}}" style="display:none;margin:6px 0 6px;padding:10px 14px;background:#0d180d;border:1px solid #253525;border-radius:4px;font-size:11px;color:#999;line-height:1.8">`;
        html += `<strong style="color:#bbb">About this analysis</strong><br>`;
        html += `For each parent system with two or more child subsystems, all child pairs (A, B) are tested. `;
        html += `Each system's activity is binarized at the median of its first principal component (PC1) across patients. `;
        html += `A majority-vote truth table is built for each (A-state, B-state) combination, then matched against `;
        html += `10 non-trivial Boolean functions (following Ma et al. 2018 DCell/VNN). A match reveals how the parent `;
        html += `system logically integrates signals from its children.<br><br>`;
        html += `<strong style="color:#bbb">Gate types:</strong><br>`;
        [['A∧B',    'AND',        'Co-requirement — both children must be active'],
         ['A∨B',    'OR',         'Redundancy — either child alone is sufficient'],
         ['A⊕B',    'XOR',        'Mutual exclusivity — exactly one child active'],
         ['¬(A∧B)', 'NAND',       'Negative synergy — blocked only when both are on'],
         ['¬(A∨B)', 'NOR',        'Dual inhibition — active only when both are off'],
         ['A↔B',    'XNOR',       'Concordance — same activation state in both children'],
         ['A∧¬B',   'A_NOT_B',   'A dominant: on when A on, B off'],
         ['B∧¬A',   'B_NOT_A',   'B dominant: on when B on, A off'],
        ].forEach(([sym, name, desc]) => {{
            html += `<span style="color:#7eb8da;font-family:monospace;display:inline-block;min-width:62px">${{sym}}</span>`;
            html += `<span style="color:#aaa"> ${{name}}</span>: ${{desc}}<br>`;
        }});
        html += `<br><strong style="color:#bbb">Consistency</strong>: fraction of patients whose parent state matches the gate prediction. `;
        html += `<span style="color:#f5a623">≥70% high</span> · <span style="color:#7eb8da">≥60% moderate</span> · <span style="color:#888">&lt;60% weak</span>`;
        html += `</div>`;
        node.bool_logic.forEach(bl => {{
            const icon      = logicIcons[bl.logic] || bl.logic;
            const desc      = logicDesc[bl.logic] || '';
            const shortDesc = desc ? desc.split(':')[0] : '';
            const pct  = Math.round(bl.consistency * 100);
            const col  = bl.consistency >= 0.7 ? '#f5a623' : bl.consistency >= 0.6 ? '#7eb8da' : '#888';
            html += `<span class="child" style="border-left:3px solid ${{col}}" `;
            html += `title="${{bl.logic}}: ${{desc}}&#10;A = ${{bl.child1}}&#10;B = ${{bl.child2}}&#10;n = ${{bl.n_samples}} samples&#10;consistency = ${{pct}}%">`;
            html += `<span style="color:${{col}};font-weight:bold">${{icon}}</span> `;
            html += `<span style="color:#888;font-size:10px">[A=${{bl.child1}}, B=${{bl.child2}}]</span> `;
            html += `<span style="color:${{col}};font-size:10px"> ${{pct}}%</span>`;
            if (shortDesc) html += ` <span style="color:#555;font-size:10px;font-style:italic">${{shortDesc}}</span>`;
            html += `</span>`;
        }});
        html += `</div>`;
    }}

    if (directGenes.length > 0) {{
        html += `<div class="children" style="margin-top:12px"><strong>Term-specific genes (${{directGenes.length}}):</strong> <span style="color:#666;font-size:11px">unique to this system, not in any child</span><br>`;
        directGenes.forEach(g => {{
            const info = ` (ρ=${{g.rho.toFixed(3)}}, p=${{g.p_val.toExponential(1)}})`;
            html += `<span class="child gene">${{g.id}}${{info}}</span>`;
        }});
        html += `</div>`;
    }}

    if (inheritedGenes.length > 0) {{
        const showCount = Math.min(inheritedGenes.length, 100);
        const label = inheritedGenes.length > showCount
            ? `Genes from child systems (${{inheritedGenes.length}}, showing top ${{showCount}} by |ρ|)`
            : `Genes from child systems (${{inheritedGenes.length}})`;
        html += `<div class="children" style="margin-top:12px"><strong>${{label}}:</strong><br>`;
        inheritedGenes.slice(0, showCount).forEach(g => {{
            const info = ` (ρ=${{g.rho.toFixed(3)}}, p=${{g.p_val.toExponential(1)}})`;
            html += `<span class="child gene">${{g.id}}${{info}}</span>`;
        }});
        html += `</div>`;
    }}

    document.getElementById('detail').innerHTML = html;
}}

document.getElementById('sort-select').addEventListener('change', renderTable);
document.getElementById('filter-input').addEventListener('input', renderTable);
document.getElementById('min-rlipp').addEventListener('input', renderTable);
renderTable();
</script>
</body>
</html>"""

    outpath.write_text(html, encoding='utf-8')


# ── Main ─────────────────────────────────────────────────────────────────────

# ── Patient-level explainability ─────────────────────────────────────────────

class PatientScoreCalculator:
    """
    Compute per-patient system and gene importance from hidden embeddings.

    Uses the .hidden files saved by predict.py (actual network activations).
    For each patient, importance is measured as how far that patient's activation
    deviates from the population mean, normalised by the population std (z-score).

      - Term importance  = ||z-score(h_i, T)||₂  (L2 norm of the 4D z-scored embedding)
      - Patient-RLIPP   = term_importance² / Σ child_importance²
      - Gene importance  = |z-score(h_i, gene)|  (signed scalar z-scored embedding)

    Using z-scores rather than raw magnitudes is necessary because BatchNorm
    normalises all activations to similar absolute magnitudes, making raw norms
    nearly identical across patients.
    """

    def __init__(self, args):
        self.hidden_dir = Path(args.hidden)
        self.num_hiddens_genotype = args.genotype_hiddens or self._detect_hiddens(self.hidden_dir)
        self.outdir = Path(args.outdir)

        ont = pd.read_csv(
            args.ontology, sep='\t', header=None,
            names=['parent', 'child', 'relation'], dtype=str
        )
        gene_leaves = set(ont[ont['relation'] == 'gene']['child'])
        all_nodes = set(ont['parent']) | set(ont['child'])
        self.terms = sorted(all_nodes - gene_leaves)
        self.genes = list(
            pd.read_csv(args.gene2id, sep='\t', header=None, names=['I', 'G'])['G']
        )

        # Derive sample order from the test/predict file (matches hidden file row order)
        self.cell_ids = self._load_cell_ids(args)

        # Predicted values for display in the viz; use sigmoid probabilities for binary tasks
        # so that patient viz percentages and thresholds are in [0,1] probability space.
        pred_path = Path(args.predicted)
        if getattr(args, 'task', 'continuous') == 'binary':
            prob_path = pred_path.parent / (pred_path.stem + '_probabilities.txt')
            if prob_path.exists():
                pred_path = prob_path
        self.predicted_vals = np.loadtxt(pred_path) if pred_path.exists() else None

        # Child term map for patient-RLIPP
        self.ont_children: dict[str, list[str]] = {}
        for _, row in ont.iterrows():
            if row['relation'] != 'gene':
                self.ont_children.setdefault(row['parent'], []).append(row['child'])

    @staticmethod
    def _detect_hiddens(hidden_dir: Path, fallback: int = 4) -> int:
        """Infer genotype_hiddens from the column count of the first readable term hidden file."""
        for path in hidden_dir.glob('*.hidden'):
            try:
                row = np.loadtxt(path, max_rows=1)
                ncols = int(row.size)
                if ncols > 1:   # gene hiddens are 1-column; skip them
                    return ncols
            except Exception:
                continue
        return fallback

    def _load_cell_ids(self, args) -> list[str]:
        test_path = Path(args.test)
        if test_path.exists():
            with open(test_path) as f:
                first = f.readline().strip()
            if 'cell_line' in first:
                df = pd.read_csv(test_path, sep='\t')
                label_col = getattr(args, 'label', None)
                if label_col and label_col in df.columns:
                    df = df.dropna(subset=[label_col])
                return list(df['cell_line'].astype(str))
            else:
                df = pd.read_csv(test_path, sep='\t', header=None)
                return list(df.iloc[:, 0].astype(str))
        # Fall back to cell2id order
        return list(
            pd.read_csv(args.cell2id, sep='\t', header=None, names=['I', 'C'])['C'].astype(str)
        )

    def _load_hidden(self, name: str, ncols: int):
        path = self.hidden_dir / f"{name}.hidden"
        if not path.exists():
            return None
        try:
            data = np.loadtxt(path)
            if data.ndim == 1:
                data = data.reshape(-1, 1)
            return data[:, :ncols]
        except Exception:
            return None

    def _zscore(self, mat: np.ndarray) -> np.ndarray:
        """Z-score each column of mat across rows (population). Returns same shape."""
        mean = mat.mean(axis=0, keepdims=True)
        std  = mat.std(axis=0, keepdims=True)
        std  = np.where(std < 1e-10, 1.0, std)
        return (mat - mean) / std

    def compute_scores(self):
        print("\nBuilding patient-level explainability ...")

        # Use training-data row count as the authoritative sample count.
        # Hidden files may have extra rows if predict.py was run multiple times
        # before the 'wb' overwrite fix; truncating here handles that.
        n_samples = len(self.cell_ids)
        cell_ids  = self.cell_ids

        term_hiddens: dict[str, np.ndarray] = {}
        for t in self.terms:
            h = self._load_hidden(t, self.num_hiddens_genotype)
            if h is not None:
                term_hiddens[t] = h[:n_samples]   # (n_samples, n_hiddens)

        gene_hiddens: dict[str, np.ndarray] = {}
        for gene in self.genes:
            h = self._load_hidden(gene, 1)
            if h is not None:
                gene_hiddens[gene] = h[:n_samples].squeeze(-1)   # (n_samples,)

        if not term_hiddens and not gene_hiddens:
            print("  No .hidden files found — skipping patient scoring.")
            print("  (Re-run predict.py to generate hidden embedding files.)")
            return None

        # Resolve any mismatch between cell_ids length and actual hidden-file rows.
        # h[:n_samples] above handles hidden files that are LONGER than cell_ids;
        # this block handles the reverse (hidden files SHORTER than cell_ids).
        actual_rows = min((h.shape[0] for h in term_hiddens.values()), default=n_samples)
        if gene_hiddens:
            actual_rows = min(actual_rows, min(h.shape[0] for h in gene_hiddens.values()))
        if actual_rows < n_samples:
            print(f"  Note: hidden files have {actual_rows} rows vs {n_samples} cell_ids — "
                  f"truncating to {actual_rows}")
            n_samples   = actual_rows
            cell_ids    = cell_ids[:n_samples]
            term_hiddens = {t: h[:n_samples] for t, h in term_hiddens.items()}
            gene_hiddens = {g: h[:n_samples] for g, h in gene_hiddens.items()}

        # Term importance: L2 norm of z-scored embedding per patient.
        # Z-scoring is required because BatchNorm makes raw norms nearly identical
        # across patients; the z-score shows deviation from the population mean.
        term_imp: dict[str, np.ndarray] = {}
        term_mean_z: dict[str, np.ndarray] = {}
        for term, h in term_hiddens.items():
            z = self._zscore(h)                          # (n_samples, n_hiddens)
            term_imp[term]    = np.linalg.norm(z, axis=1) # (n_samples,) unsigned
            term_mean_z[term] = z.mean(axis=1)            # (n_samples,) signed

        # Patient-RLIPP: norm²(parent) / Σ norm²(children)
        patient_rlipp: dict[str, np.ndarray] = {}
        for term, imp in term_imp.items():
            children = self.ont_children.get(term, [])
            child_imps = [term_imp[c] for c in children if c in term_imp]
            if not child_imps:
                continue
            child_sq = sum(c ** 2 for c in child_imps)
            rlipp = np.zeros(len(imp))
            mask = child_sq > 1e-10
            rlipp[mask] = (imp[mask] ** 2) / child_sq[mask]
            patient_rlipp[term] = rlipp

        # Gene importance: absolute z-score of the scalar gene embedding
        gene_imp: dict[str, np.ndarray] = {}
        gene_signed_z: dict[str, np.ndarray] = {}
        for gene, h in gene_hiddens.items():
            mean, std = h.mean(), h.std()
            std = std if std > 1e-10 else 1.0
            z = (h - mean) / std
            gene_signed_z[gene] = z
            gene_imp[gene] = np.abs(z)

        print(f"  {len(term_imp)}/{len(self.terms)} terms, "
              f"{len(gene_imp)}/{len(self.genes)} genes, {n_samples} samples")

        # Write TSV outputs
        if term_imp:
            pd.DataFrame(term_imp, index=cell_ids).rename_axis('cell_id').to_csv(
                self.outdir / 'patient_term_importance.txt', sep='\t', float_format='%.4f', encoding='utf-8'
            )
        if patient_rlipp:
            pd.DataFrame(patient_rlipp, index=cell_ids).rename_axis('cell_id').to_csv(
                self.outdir / 'patient_rlipp.txt', sep='\t', float_format='%.4f', encoding='utf-8'
            )
        if gene_imp:
            pd.DataFrame(gene_imp, index=cell_ids).rename_axis('cell_id').to_csv(
                self.outdir / 'patient_gene_importance.txt', sep='\t', float_format='%.4f', encoding='utf-8'
            )
        if gene_signed_z:
            pd.DataFrame(gene_signed_z, index=cell_ids).rename_axis('cell_id').to_csv(
                self.outdir / 'patient_gene_signed_z.txt', sep='\t', float_format='%.4f', encoding='utf-8'
            )
        if term_mean_z:
            pd.DataFrame(term_mean_z, index=cell_ids).rename_axis('cell_id').to_csv(
                self.outdir / 'patient_term_mean_z.txt', sep='\t', float_format='%.4f', encoding='utf-8'
            )

        return term_imp, patient_rlipp, gene_imp, gene_signed_z, term_mean_z, cell_ids


def build_patient_viz(term_imp, patient_rlipp, gene_imp, gene_signed_z, term_mean_z, cell_ids,
                      predicted_vals, pop_rlipp_df, pop_gene_df, outpath,
                      study_id=None, task='continuous', label='score'):
    """Build a standalone interactive HTML for per-patient system/gene importance."""
    import math, json

    def r4(v):
        try:
            f = float(v)
            return 0.0 if (math.isnan(f) or math.isinf(f)) else round(f, 4)
        except (TypeError, ValueError):
            return 0.0

    n = len(cell_ids)

    # Compact JSON: {term: [val_p0, val_p1, ...]}  (rounded to 4 dp)
    term_imp_j    = json.dumps({t: [r4(v) for v in vals] for t, vals in term_imp.items()})
    pt_rlipp_j    = json.dumps({t: [r4(v) for v in vals] for t, vals in patient_rlipp.items()})
    term_mz_j     = json.dumps({t: [r4(v) for v in vals] for t, vals in term_mean_z.items()})
    gene_imp_j    = json.dumps({g: [r4(v) for v in vals] for g, vals in gene_imp.items()})
    gene_sz_j     = json.dumps({g: [r4(v) for v in vals] for g, vals in gene_signed_z.items()})
    cell_ids_j    = json.dumps(list(cell_ids))
    preds_j       = json.dumps([r4(v) for v in predicted_vals[:n]] if predicted_vals is not None else [None] * n)
    study_id_j    = json.dumps(study_id)
    task_j        = json.dumps(task)
    label_j       = json.dumps(label)
    pop_rlipp_j   = json.dumps(
        {row['term']: r4(row['rlipp']) for _, row in pop_rlipp_df.iterrows()}
        if pop_rlipp_df is not None and not pop_rlipp_df.empty else {}
    )
    pop_term_prho_j = json.dumps(
        {row['term']: r4(row['p_rho']) for _, row in pop_rlipp_df.iterrows()}
        if pop_rlipp_df is not None and not pop_rlipp_df.empty else {}
    )
    pop_gene_rho_j = json.dumps(
        {row['gene']: r4(row['rho']) for _, row in pop_gene_df.iterrows()}
        if pop_gene_df is not None and not pop_gene_df.empty else {}
    )

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>NeST-VNN Patient Explainability</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ background: #0e0e0e; color: #ddd; font-family: monospace; font-size: 13px;
       height: 100vh; display: flex; flex-direction: column; overflow: hidden; }}
#header {{ padding: 8px 16px; background: #141414; border-bottom: 1px solid #222;
           display: flex; align-items: center; gap: 20px; flex-shrink: 0; }}
#header h1 {{ font-size: 14px; color: #7eb8da; }}
.stat {{ color: #888; font-size: 12px; }}
#controls {{ padding: 8px 16px; background: #111; border-bottom: 1px solid #1a1a1a;
             display: flex; align-items: center; gap: 12px; flex-shrink: 0; flex-wrap: wrap; }}
#controls label {{ color: #888; font-size: 12px; }}
#patient-search {{ background: #1a1a2e; border: 1px solid #333; color: #ddd;
                   padding: 4px 8px; font-family: monospace; font-size: 12px;
                   width: 240px; border-radius: 3px; }}
#patient-select {{ background: #1a1a2e; border: 1px solid #333; color: #ddd;
                   padding: 4px 8px; font-family: monospace; font-size: 12px;
                   width: 240px; border-radius: 3px; }}
#pred-info {{ color: #f5a623; font-size: 12px; }}
#cbio-link {{ color: #7eb8da; font-size: 12px; text-decoration: none; border: 1px solid #2a4a6a;
              border-radius: 3px; padding: 3px 8px; white-space: nowrap; }}
#cbio-link:hover {{ background: #1a2a3a; }}
#cbio-link.hidden {{ display: none; }}
#main {{ display: flex; flex: 1; overflow: hidden; }}
.panel {{ flex: 1; display: flex; flex-direction: column; border-right: 1px solid #1a1a1a;
          min-width: 0; }}
.panel:last-child {{ border-right: none; }}
.panel-title {{ padding: 6px 10px; background: #111; color: #888; font-size: 11px;
                text-transform: uppercase; letter-spacing: 0.5px; border-bottom: 1px solid #1a1a1a;
                flex-shrink: 0; }}
.panel-scroll {{ flex: 1; overflow-y: auto; }}
table {{ width: 100%; border-collapse: collapse; }}
th {{ text-align: left; padding: 5px 8px; border-bottom: 1px solid #2a2a2a;
      position: sticky; top: 0; background: #0e0e0e; color: #777; font-size: 11px; }}
td {{ padding: 4px 8px; border-bottom: 1px solid #141414; white-space: nowrap; }}
tr:hover td {{ background: #1a2a3a; }}
.bar {{ display: inline-block; height: 9px; border-radius: 2px; min-width: 1px; vertical-align: middle; }}
.rlipp-high {{ color: #f5a623; font-weight: bold; }}
.rlipp-mid  {{ color: #7eb8da; }}
.rlipp-low  {{ color: #555; }}
.note {{ padding: 16px; color: #555; font-size: 12px; }}
#interp-bar {{ padding: 8px 16px; background: #0b0b0b; border-bottom: 1px solid #1a1a1a;
               display: flex; gap: 24px; align-items: center; flex-shrink: 0; flex-wrap: wrap;
               font-size: 12px; color: #888; min-height: 32px; }}
.dir-up {{ color: #e08030; font-weight: bold; }}
.dir-dn {{ color: #50a870; font-weight: bold; }}
</style>
</head>
<body>
<div id="header">
  <h1>NeST-VNN · Patient-Level Explainability</h1>
  <span class="stat" id="s-patients"></span>
  <span class="stat" id="s-terms"></span>
  <span class="stat" id="s-genes"></span>
</div>
<div id="controls">
  <label>Patient:</label>
  <input id="patient-search" type="text" placeholder="Search ID …">
  <select id="patient-select"></select>
  <span id="pred-info"></span>
  <a id="cbio-link" href="#" target="_blank" class="hidden">↗ cBioPortal</a>
</div>
<div id="interp-bar">
  <span id="interp-pred"></span>
  <span id="interp-top-sys"></span>
  <span id="interp-top-gene"></span>
</div>
<div id="main">
  <div class="panel">
    <div class="panel-title">
      Systems · ranked by patient importance
      <span style="color:#555;font-style:normal;text-transform:none;font-size:10px">
        &nbsp;Importance = ||z-score of hidden embedding||  &nbsp;|&nbsp;  Pt-RLIPP = imp²/Σchild²  &nbsp;|&nbsp;  Pop-RLIPP = cross-cohort
      </span>
    </div>
    <div class="panel-scroll">
      <table>
        <thead><tr>
          <th>System</th>
          <th title="L2 norm of z-scored hidden embedding: how far this patient's system activation deviates from the population mean">Importance</th>
          <th title="Patient-RLIPP: deviation²(parent) / Σdeviation²(children) — how much this system's anomaly exceeds its children's">Pt-RLIPP</th>
          <th title="Population-level RLIPP from cross-cohort analysis">Pop-RLIPP</th>
          <th title="Direction: sign(mean z-score across hidden dims) × sign(pop p_rho). Heuristic — see README.">Outcome Dir</th>
          <th></th>
        </tr></thead>
        <tbody id="sys-body"></tbody>
      </table>
    </div>
  </div>
  <div class="panel">
    <div class="panel-title">Genes · ranked by patient importance (top 100)</div>
    <div class="panel-scroll">
      <table>
        <thead><tr>
          <th>Gene</th>
          <th title="|z-score| of gene hidden embedding: how far this patient's gene activation deviates from the population">Importance</th>
          <th title="Signed z-score: positive = above population mean, negative = below">Signed Z</th>
          <th title="Direction of this gene's deviation relative to predicted outcome (sign of z × sign of cohort ρ)">Outcome Dir</th>
          <th></th>
        </tr></thead>
        <tbody id="gene-body"></tbody>
      </table>
    </div>
  </div>
</div>
<script>
const cellIds      = {cell_ids_j};
const preds        = {preds_j};
const termImp      = {term_imp_j};
const ptRlipp      = {pt_rlipp_j};
const termMZ       = {term_mz_j};
const geneImp      = {gene_imp_j};
const geneSZ       = {gene_sz_j};
const popRlipp     = {pop_rlipp_j};
const popTermPrho  = {pop_term_prho_j};
const popGeneRho   = {pop_gene_rho_j};
const studyId      = {study_id_j};
const task         = {task_j};
const labelName    = {label_j};

const termList = Object.keys(termImp);
const geneList = Object.keys(geneImp);
const N = cellIds.length;

document.getElementById('s-patients').textContent = N + ' patients';
document.getElementById('s-terms').textContent    = termList.length + ' systems';
document.getElementById('s-genes').textContent    = geneList.length + ' genes';

// Build patient selector
const sel = document.getElementById('patient-select');
function rebuildSelect(ids) {{
    sel.innerHTML = '';
    ids.forEach(({{id, idx}}) => {{
        const o = document.createElement('option');
        o.value = idx; o.text = id;
        sel.appendChild(o);
    }});
}}
const allPatients = cellIds.map((id, idx) => ({{id, idx}}));
rebuildSelect(allPatients);

function rlippClass(v) {{
    return v > 1.2 ? 'rlipp-high' : v > 1.0 ? 'rlipp-mid' : 'rlipp-low';
}}

// Direction helpers — sign(patient deviation) × sign(cohort ρ / p_rho)
// +1 → deviation pushes toward higher predicted score
// -1 → deviation pushes toward lower predicted score
//  0 → signal too weak to call
function geneDir(gene, patIdx) {{
    const sz  = (geneSZ[gene]    || [])[patIdx] ?? 0;
    const rho = popGeneRho[gene] ?? 0;
    if (Math.abs(sz) < 0.5 || Math.abs(rho) < 0.1) return 0;
    return Math.sign(sz) * Math.sign(rho);
}}

// Term direction: sign(mean z-score across hidden dims) × sign(pop p_rho)
// Mean z is a signed summary of the multi-dimensional embedding deviation.
function termDir(term, patIdx) {{
    const mz  = (termMZ[term]      || [])[patIdx] ?? 0;
    const rho = popTermPrho[term]  ?? 0;
    if (Math.abs(mz) < 0.2 || Math.abs(rho) < 0.1) return 0;
    return Math.sign(mz) * Math.sign(rho);
}}

function dirLabel(dir) {{
    if (dir > 0) return '<span class="dir-up" title="Deviation in this patient is associated with higher predicted score">↑ higher</span>';
    if (dir < 0) return '<span class="dir-dn" title="Deviation in this patient is associated with lower predicted score">↓ lower</span>';
    return '<span style="color:#444">—</span>';
}}

function buildInterp(patIdx) {{
    const pred = preds[patIdx];

    // 1. Prediction summary
    let predStr = '';
    if (pred !== null && pred !== undefined) {{
        if (task === 'binary') {{
            const pct = Math.round(pred * 100);
            const lvl = pred >= 0.7 ? 'high' : pred >= 0.5 ? 'moderate-high'
                      : pred >= 0.3 ? 'moderate-low' : 'low';
            predStr = `Pred: <strong style="color:#f5a623">${{pct}}%</strong> — ${{lvl}} ${{labelName}}`;
        }} else {{
            const sign = pred >= 0 ? '+' : '';
            const dir  = pred >= 0 ? 'above' : 'below';
            predStr = `Pred: <strong style="color:#f5a623">${{sign}}${{pred.toFixed(3)}}</strong> (${{dir}} cohort mean)`;
        }}
    }}

    // 2. Top system (highest patient importance)
    const tRows = Object.keys(termImp)
        .map(t => ({{ id: t, imp: (termImp[t] || [])[patIdx] ?? 0, popR: popRlipp[t] ?? null }}))
        .sort((a, b) => b.imp - a.imp);
    let sysStr = '';
    if (tRows.length) {{
        const top = tRows[0];
        let extra = '';
        if (top.popR !== null) {{
            extra = `, pop-RLIPP=${{top.popR.toFixed(2)}}`;
            if (top.popR > 1.2) extra += ' <span style="color:#f5a623">✓ cohort-validated</span>';
        }}
        const tDir = termDir(top.id, patIdx);
        const tDirSpan = tDir !== 0 ? ' → ' + (tDir > 0
            ? `<span class="dir-up">↑ higher</span>`
            : `<span class="dir-dn">↓ lower</span>`) + ` ${{labelName}}` : '';
        sysStr = `Top system: <strong>${{top.id}}</strong> (imp=${{top.imp.toFixed(2)}}${{extra}})${{tDirSpan}}`;
    }}

    // 3. Top gene with a clear directional signal
    let geneStr = '';
    const gRows = Object.keys(geneImp)
        .map(g => ({{ id: g, imp: (geneImp[g] || [])[patIdx] ?? 0, sz: (geneSZ[g] || [])[patIdx] ?? 0 }}))
        .sort((a, b) => b.imp - a.imp);
    for (const g of gRows.slice(0, 30)) {{
        const rho = popGeneRho[g.id] ?? 0;
        if (Math.abs(g.sz) >= 0.8 && Math.abs(rho) >= 0.1) {{
            const dir    = Math.sign(g.sz) * Math.sign(rho);
            const szSign = g.sz >= 0 ? '+' : '';
            const rhoSign = rho >= 0 ? '+' : '';
            const dirSpan = dir > 0
                ? '<span class="dir-up">↑ higher</span>'
                : '<span class="dir-dn">↓ lower</span>';
            geneStr = `Top gene: <strong>${{g.id}}</strong> (z=${{szSign}}${{g.sz.toFixed(2)}}, cohort ρ=${{rhoSign}}${{rho.toFixed(2)}}) → ${{dirSpan}} ${{labelName}}`;
            break;
        }}
    }}
    if (!geneStr && gRows.length) {{
        const g = gRows[0];
        const szSign = g.sz >= 0 ? '+' : '';
        geneStr = `Top gene: <strong>${{g.id}}</strong> (z=${{szSign}}${{g.sz.toFixed(2)}}, no clear directional signal)`;
    }}

    document.getElementById('interp-pred').innerHTML     = predStr;
    document.getElementById('interp-top-sys').innerHTML  = sysStr;
    document.getElementById('interp-top-gene').innerHTML = geneStr;
}}

function render(patIdx) {{
    const pred = preds[patIdx];
    const info = document.getElementById('pred-info');
    info.textContent = pred !== null && pred !== undefined
        ? 'Prediction: ' + pred.toFixed(4) : '';
    buildInterp(patIdx);

    const cbioLink = document.getElementById('cbio-link');
    if (studyId && cellIds[patIdx]) {{
        cbioLink.href = 'https://www.cbioportal.org/patient?sampleId=' +
            encodeURIComponent(cellIds[patIdx]) + '&studyId=' + encodeURIComponent(studyId);
        cbioLink.classList.remove('hidden');
    }} else {{
        cbioLink.classList.add('hidden');
    }}

    // Systems
    const termRows = termList.map(t => ({{
        id:      t,
        imp:     (termImp[t]  || [])[patIdx] ?? 0,
        ptR:     (ptRlipp[t]  || [])[patIdx] ?? null,
        popR:    popRlipp[t]  ?? null,
    }})).sort((a, b) => b.imp - a.imp);

    const maxImp = termRows.length ? termRows[0].imp || 1 : 1;
    document.getElementById('sys-body').innerHTML = termRows.map(s => {{
        const w   = Math.max(1, Math.min(80, (s.imp / maxImp) * 80));
        const ptS = s.ptR !== null
            ? `<span class="${{rlippClass(s.ptR)}}">${{s.ptR.toFixed(3)}}</span>` : '—';
        const ppS = s.popR !== null
            ? `<span class="${{rlippClass(s.popR)}}">${{s.popR.toFixed(3)}}</span>` : '—';
        const dir = termDir(s.id, patIdx);
        return `<tr><td>${{s.id}}</td><td>${{s.imp.toFixed(4)}}</td>
            <td>${{ptS}}</td><td>${{ppS}}</td>
            <td>${{dirLabel(dir)}}</td>
            <td><span class="bar" style="width:${{w}}px;background:#3a6ea8"></span></td></tr>`;
    }}).join('');

    // Genes (top 100 by importance for this patient)
    const geneRows = geneList.map(g => ({{
        id:  g,
        imp: (geneImp[g] || [])[patIdx] ?? 0,
        sz:  (geneSZ[g]  || [])[patIdx] ?? 0,
    }})).sort((a, b) => b.imp - a.imp).slice(0, 100);

    const maxGImp = geneRows.length ? geneRows[0].imp || 1 : 1;
    document.getElementById('gene-body').innerHTML = geneRows.map(g => {{
        const w      = Math.max(1, Math.min(80, (g.imp / maxGImp) * 80));
        const dir    = geneDir(g.id, patIdx);
        const szSign = g.sz >= 0 ? '+' : '';
        const szColor = Math.abs(g.sz) >= 1.5 ? '#ddd' : Math.abs(g.sz) >= 0.8 ? '#999' : '#555';
        return `<tr><td>${{g.id}}</td><td>${{g.imp.toFixed(4)}}</td>
            <td style="color:${{szColor}}">${{szSign}}${{g.sz.toFixed(3)}}</td>
            <td>${{dirLabel(dir)}}</td>
            <td><span class="bar" style="width:${{w}}px;background:#3a7a50"></span></td></tr>`;
    }}).join('');
}}

document.getElementById('patient-search').addEventListener('input', function() {{
    const q = this.value.toLowerCase();
    const filtered = allPatients.filter(p => p.id.toLowerCase().includes(q));
    rebuildSelect(filtered);
    if (filtered.length) render(filtered[0].idx);
}});
sel.addEventListener('change', () => render(parseInt(sel.value)));

if (N > 0) render(allPatients[0].idx);
</script>
</body>
</html>"""

    outpath.write_text(html, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(
        description='NeST-VNN Explainability: RLIPP + Annotated Hierarchy'
    )
    parser.add_argument('study_id', help='cBioPortal study ID (e.g. laml_tcga_pub)')
    parser.add_argument('--data-dir', default='data', help='Base data directory')
    parser.add_argument('-hidden', default=None, help='Hidden embeddings directory (override)')
    parser.add_argument('-predicted', default=None, help='Predicted values file (override)')
    parser.add_argument('-test', default=None, help='Test data file (override)')
    parser.add_argument('-label', default=None, help='Label column (for new-format test files)')
    parser.add_argument('-task', default='continuous', choices=['continuous', 'binary'])
    parser.add_argument('-cpu_count', type=int, default=1, help='CPU cores for parallel computation')
    parser.add_argument('-genotype_hiddens', type=int, default=None, help='Hidden dim per term (auto-detected from hidden files if omitted)')
    parser.add_argument('-mlflow', action='store_true', help='Log annotation artifacts to predict MLflow run')

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    study_dir = data_dir / "output" / args.study_id
    input_dir = study_dir / "nest_vnn_input"
    label_dir = args.label if args.label else "default"
    run_dir = study_dir / label_dir
    metrics_dir = run_dir / "metrics"
    annotation_dir = run_dir / "annotation"
    annotation_dir.mkdir(parents=True, exist_ok=True)

    # Resolve paths from convention, allow overrides
    args.ontology = str(input_dir / "ontology.txt")
    args.gene2id = str(input_dir / "gene2ind.txt")
    args.cell2id = str(input_dir / "cell2ind.txt")
    args.outdir = str(annotation_dir)

    if args.hidden is None:
        args.hidden = str(metrics_dir / "hidden")
    if args.predicted is None:
        args.predicted = str(metrics_dir / "predict.txt")
    if args.test is None:
        args.test = str(input_dir / "training_data.txt")

    print("=" * 60)
    print("NeST-VNN Explainability Pipeline")
    print("=" * 60)
    print(f"Study:      {args.study_id}")
    print(f"Label:      {label_dir}")
    print(f"Input:      {input_dir}")
    print(f"Run dir:    {run_dir}")
    print(f"Annotation: {annotation_dir}\n")

    # Compute RLIPP, gene scores, subsystem gene weights, and boolean logic
    calculator = ClinicalRLIPPCalculator(args)
    rlipp_df, gene_df, subsys_gene_df, bool_logic_df = calculator.calc_scores()

    # Build annotated hierarchy
    print("\nBuilding annotated hierarchy ...")
    outdir = Path(args.outdir)
    build_annotated_hierarchy(args.ontology, rlipp_df, gene_df, outdir,
                              subsys_gene_df=subsys_gene_df,
                              bool_logic_df=bool_logic_df)

    # Detect cBioPortal study for patient page links
    meta_path = input_dir / "metadata.json"
    cbio_study_id = None
    if meta_path.exists():
        import json as _json
        meta = _json.loads(meta_path.read_text())
        if meta.get("cbioportal_url"):
            cbio_study_id = meta.get("study_id", args.study_id)

    # Patient-level explainability
    patient_calc = PatientScoreCalculator(args)
    patient_result = patient_calc.compute_scores()
    if patient_result is not None:
        term_imp, patient_rlipp_scores, gene_imp, gene_signed_z, term_mean_z, cell_ids_used = patient_result
        patient_html = outdir / 'patient_viz.html'
        build_patient_viz(
            term_imp, patient_rlipp_scores, gene_imp, gene_signed_z, term_mean_z, cell_ids_used,
            patient_calc.predicted_vals, rlipp_df, gene_df, patient_html,
            study_id=cbio_study_id,
            task=args.task,
            label=args.label or 'score',
        )
        print(f"  patient_viz.html → {patient_html}")

    print(f"\n{'='*60}")
    print(f"DONE. Outputs in: {outdir.resolve()}")
    print(f"{'='*60}")
    print(f"""
Output files:
  rlipp_scores.txt            — RLIPP scores for all ontology terms (cross-cohort)
  gene_scores.txt             — Gene-level Spearman correlations (cross-cohort)
  subsystem_gene_weights.txt  — Per-term gene rankings by PC1 correlation (all terms)
  top_subsystem_genes.txt     — Top 20 terms × top 5 driving genes (quick-read summary)
  boolean_logic.txt           — Boolean logic characterization of subsystem trios (AND/OR/XOR/etc.)
  hierarchy_annotated.graphml — Annotated hierarchy (open in Cytoscape)
  hierarchy_annotated.cx2     — Annotated hierarchy in CX2 format (NDEx/Cytoscape Web)
  hierarchy_viz.html          — Interactive HTML visualization (open in browser)
  top_systems.txt             — Top 20 systems by RLIPP
  patient_term_importance.txt — Per-patient term importance scores (z-scored hidden embedding L2 norms)
  patient_rlipp.txt           — Per-patient RLIPP scores
  patient_gene_importance.txt — Per-patient gene importance scores
  patient_viz.html            — Interactive per-patient explainability viewer

The RLIPP score measures relative local improvement in predictive power:
  RLIPP > 1: the system's hidden representation adds information
             beyond its children (important system)
  RLIPP ≈ 1: the system doesn't add much beyond its children
  RLIPP < 1: children are more informative than the parent
""")

    annotation_files = [
        "rlipp_scores.txt", "gene_scores.txt",
        "subsystem_gene_weights.txt", "top_subsystem_genes.txt", "boolean_logic.txt",
        "hierarchy_annotated.graphml", "hierarchy_viz.html",
        "top_systems.txt", "hierarchy_annotated.cx2",
        "patient_term_importance.txt", "patient_rlipp.txt",
        "patient_gene_importance.txt", "patient_viz.html",
    ]

    if args.mlflow:
        try:
            import mlflow
            run_id_path = metrics_dir / "mlflow_run_id.txt"
            predict_run_id = run_id_path.read_text().strip() if run_id_path.exists() else None
            run_kwargs = {"run_id": predict_run_id} if predict_run_id else {}
            with mlflow.start_run(**run_kwargs):
                for fname in annotation_files:
                    fpath = outdir / fname
                    if fpath.exists():
                        mlflow.log_artifact(str(fpath), artifact_path="annotation")
        except ImportError:
            print("Warning: mlflow not installed; skipping MLflow artifact logging.")


if __name__ == "__main__":
    main()