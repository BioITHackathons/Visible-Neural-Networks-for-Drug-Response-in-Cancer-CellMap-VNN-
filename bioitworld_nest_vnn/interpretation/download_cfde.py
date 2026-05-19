"""Download the CFDE IDG Druggable Genome catalog from Pharos/TCRD."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from interpretation.cfde_drugs import download, CACHE_PATH

if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else CACHE_PATH
    lookup, tdl_counts = download(out_path=out)

    total = sum(tdl_counts.values())
    print(f"\n{'='*55}")
    print(f"CFDE IDG TDL classification for our {total} model genes:")
    for tdl in ["Tclin", "Tchem", "Tbio", "Tdark", "unknown"]:
        n = tdl_counts.get(tdl, 0)
        if n:
            bar = "█" * (n * 30 // total)
            print(f"  {tdl:<8} {n:>4}  {bar}")
    print(f"\nTclin genes with drug entries: {len(lookup)}")
    print("Sample Tclin entries:")
    for gene, drugs in list(lookup.items())[:8]:
        print(f"  {gene}: {drugs[:4]}")
