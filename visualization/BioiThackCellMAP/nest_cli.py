#!/usr/bin/env python3
"""
NeST-STRING CLI
Loads a NeST CX network file, lets you browse NEST groups, and queries STRING API.

Usage:
    python3 nest_cli.py <path_to_cx_file>
    python3 nest_cli.py <path_to_cx_file> --group NEST:129
    python3 nest_cli.py <path_to_cx_file> --group NEST:92 --enrichment
"""

import json
import sys
import argparse
import requests

STRING_BASE = "https://string-db.org/api/json"
SPECIES_HUMAN = 9606


# ── CX parser ────────────────────────────────────────────────────────────────

def parse_cx(path: str) -> dict:
    with open(path) as f:
        data = json.load(f)

    nodes = {}
    edges = []
    node_attrs = {}

    for item in data:
        key = list(item.keys())[0]
        val = item[key]

        if key == "nodes":
            for n in val:
                nodes[n["@id"]] = {"name": n["n"], "attrs": {}}

        elif key == "edges":
            for e in val:
                edges.append({"id": e["@id"], "source": e["s"], "target": e["t"]})

        elif key == "nodeAttributes":
            for a in val:
                nid = a["po"]
                if nid not in node_attrs:
                    node_attrs[nid] = {}
                node_attrs[nid][a["n"]] = a["v"]

    for nid, attrs in node_attrs.items():
        if nid in nodes:
            nodes[nid]["attrs"] = attrs

    # Build name→id index and adjacency
    name_to_id = {n["name"]: nid for nid, n in nodes.items()}
    children: dict[str, list[str]] = {n["name"]: [] for n in nodes.values()}
    parents: dict[str, list[str]] = {n["name"]: [] for n in nodes.values()}

    for e in edges:
        src = nodes.get(e["source"], {}).get("name", "?")
        tgt = nodes.get(e["target"], {}).get("name", "?")
        children[src].append(tgt)
        parents[tgt].append(src)

    return {
        "nodes": nodes,
        "name_to_id": name_to_id,
        "children": children,
        "parents": parents,
    }


# ── Display helpers ───────────────────────────────────────────────────────────

def get_genes(cx: dict, group_name: str) -> list[str]:
    nid = cx["name_to_id"].get(group_name)
    if nid is None:
        return []
    raw = cx["nodes"][nid]["attrs"].get("Genes", "")
    return raw.split() if raw else []


def print_tree(cx: dict, root: str, indent: int = 0):
    nid = cx["name_to_id"].get(root)
    if nid is None:
        return
    attrs = cx["nodes"][nid]["attrs"]
    size = attrs.get("Size", "?")
    annot = attrs.get("Annotation", "")
    prefix = "  " * indent + ("└─ " if indent else "")
    print(f"{prefix}{root}  [{size} genes]  {annot}")
    for child in cx["children"].get(root, []):
        print_tree(cx, child, indent + 1)


def print_all_groups(cx: dict):
    # Find root nodes (no parents)
    roots = [name for name, plist in cx["parents"].items() if not plist]
    print("\n=== NeST Group Hierarchy ===\n")
    for root in sorted(roots):
        print_tree(cx, root)

    # Print flat list with genes
    print("\n=== All Groups (flat) ===\n")
    print(f"{'Group':<12} {'Size':>5}  {'Genes'}")
    print("-" * 80)
    for name in sorted(cx["name_to_id"].keys()):
        nid = cx["name_to_id"][name]
        attrs = cx["nodes"][nid]["attrs"]
        size = attrs.get("Size", "?")
        genes = attrs.get("Genes", "")
        print(f"{name:<12} {size:>5}  {genes}")


# ── STRING queries ────────────────────────────────────────────────────────────

def query_string_network(genes: list[str], required_score: int = 400) -> list[dict]:
    payload = {
        "identifiers": "%0d".join(genes),
        "species": SPECIES_HUMAN,
        "required_score": required_score,
        "caller_identity": "nest_string_tool",
    }
    r = requests.post(f"{STRING_BASE}/network", data=payload, timeout=30)
    r.raise_for_status()
    return r.json()


def query_string_enrichment(genes: list[str]) -> list[dict]:
    payload = {
        "identifiers": "%0d".join(genes),
        "species": SPECIES_HUMAN,
        "caller_identity": "nest_string_tool",
    }
    r = requests.post(f"{STRING_BASE}/enrichment", data=payload, timeout=60)
    r.raise_for_status()
    return r.json()


# ── Output formatters ─────────────────────────────────────────────────────────

def print_interactions(interactions: list[dict], top: int = 20):
    if not interactions:
        print("  No interactions found.")
        return
    print(f"\n=== Protein-Protein Interactions (top {min(top, len(interactions))}) ===\n")
    print(f"{'ProteinA':<15} {'ProteinB':<15} {'Score':>8}  {'Coexpression':>14}  {'Experimental':>13}")
    print("-" * 75)
    sorted_ppis = sorted(interactions, key=lambda x: float(x.get("score", 0)), reverse=True)
    for ppi in sorted_ppis[:top]:
        a = ppi.get("preferredName_A", ppi.get("stringId_A", "?"))
        b = ppi.get("preferredName_B", ppi.get("stringId_B", "?"))
        score = float(ppi.get("score", 0))
        coexp = float(ppi.get("coexpression", 0))
        exp = float(ppi.get("experimentally_determined_interaction", 0))
        print(f"{a:<15} {b:<15} {score:>8.3f}  {coexp:>14.3f}  {exp:>13.3f}")
    if len(interactions) > top:
        print(f"  ... and {len(interactions) - top} more interactions")


def print_enrichment(terms: list[dict], top: int = 20):
    if not terms:
        print("  No enrichment terms found.")
        return
    print(f"\n=== Functional Enrichment Analysis (top {min(top, len(terms))} by p-value) ===\n")
    print(f"{'Category':<12} {'Term':<45} {'FDR':>12}  Genes")
    print("-" * 100)
    sorted_terms = sorted(terms, key=lambda x: float(x.get("fdr", 1)))
    for t in sorted_terms[:top]:
        cat = t.get("category", "?")[:12]
        desc = t.get("description", "?")[:45]
        fdr = float(t.get("fdr", 1))
        gene_list = ",".join(t.get("inputGenes", [])[:6])
        if len(t.get("inputGenes", [])) > 6:
            gene_list += "..."
        print(f"{cat:<12} {desc:<45} {fdr:>12.2e}  {gene_list}")
    if len(terms) > top:
        print(f"  ... and {len(terms) - top} more terms")


# ── Interactive selector ──────────────────────────────────────────────────────

def interactive_select(cx: dict) -> str:
    groups = sorted(cx["name_to_id"].keys())
    print("\nAvailable NEST groups:")
    for i, g in enumerate(groups):
        nid = cx["name_to_id"][g]
        size = cx["nodes"][nid]["attrs"].get("Size", "?")
        annot = cx["nodes"][nid]["attrs"].get("Annotation", "")
        print(f"  [{i+1}] {g:12s}  size={size:>3}  {annot}")
    print()
    while True:
        choice = input("Enter group name (e.g. NEST:92) or number: ").strip()
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(groups):
                return groups[idx]
        elif choice in cx["name_to_id"]:
            return choice
        print("  Invalid choice, try again.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="NeST-STRING CLI: explore NeST groups and query STRING API"
    )
    parser.add_argument("cx_file", help="Path to the .cx network file")
    parser.add_argument("--group", help="NEST group ID to query (e.g. NEST:129)")
    parser.add_argument("--enrichment", action="store_true", help="Also run enrichment analysis")
    parser.add_argument("--score", type=int, default=400, help="Minimum STRING score (0-1000, default 400)")
    parser.add_argument("--top", type=int, default=20, help="Number of results to display")
    args = parser.parse_args()

    print(f"Loading {args.cx_file} ...")
    cx = parse_cx(args.cx_file)
    print(f"Loaded {len(cx['nodes'])} NeST groups.\n")

    print_all_groups(cx)

    # Resolve group
    if args.group:
        group = args.group
        if group not in cx["name_to_id"]:
            print(f"Error: group '{group}' not found in file.")
            sys.exit(1)
    else:
        group = interactive_select(cx)

    genes = get_genes(cx, group)
    if not genes:
        print(f"No genes found for {group}.")
        sys.exit(1)

    nid = cx["name_to_id"][group]
    annot = cx["nodes"][nid]["attrs"].get("Annotation", "")
    print(f"\nSelected: {group}  ({annot})")
    print(f"Genes ({len(genes)}): {' '.join(genes)}\n")

    # STRING network
    print(f"Querying STRING for PPI network (min score={args.score}) ...")
    try:
        interactions = query_string_network(genes, required_score=args.score)
        print_interactions(interactions, top=args.top)
    except requests.RequestException as e:
        print(f"STRING network query failed: {e}")

    # Enrichment
    if args.enrichment:
        print(f"\nQuerying STRING for enrichment analysis ...")
        try:
            terms = query_string_enrichment(genes)
            print_enrichment(terms, top=args.top)
        except requests.RequestException as e:
            print(f"STRING enrichment query failed: {e}")


if __name__ == "__main__":
    main()
