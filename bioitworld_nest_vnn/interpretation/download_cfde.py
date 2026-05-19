"""Download the CFDE Druggable Genome catalog from Pharos and cache locally."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from interpretation.cfde_drugs import download, CACHE_PATH

if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else CACHE_PATH
    lookup = download(out_path=out)

    print(f"\n{'='*50}")
    print(f"Total genes with approved CFDE drugs: {len(lookup)}")
    print("Sample entries:")
    for gene, drugs in list(lookup.items())[:8]:
        print(f"  {gene}: {drugs[:4]}")
