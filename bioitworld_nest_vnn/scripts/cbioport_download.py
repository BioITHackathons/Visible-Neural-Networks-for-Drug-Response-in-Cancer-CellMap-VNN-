"""
Download mutation, fusion, CNV, and clinical data from cBioPortal for any study.

Usage:
    python cbioportal_download.py <study_id>
    python cbioportal_download.py laml_tcga_pub
    python cbioportal_download.py breast_msk_2025
"""

import argparse
import requests
import pandas as pd
import sys
from pathlib import Path
from datetime import datetime

BASE_URL = "https://www.cbioportal.org/api"
HEADERS = {"Accept": "application/json"}


# ── API helpers ──────────────────────────────────────────────────────────────

def api_get(endpoint: str, params: dict | None = None) -> list | dict:
    url = f"{BASE_URL}/{endpoint}"
    resp = requests.get(url, headers=HEADERS, params=params or {})
    resp.raise_for_status()
    return resp.json()


def api_post(endpoint: str, payload: dict) -> list:
    url = f"{BASE_URL}/{endpoint}"
    resp = requests.post(
        url,
        headers={**HEADERS, "Content-Type": "application/json"},
        json=payload,
    )
    resp.raise_for_status()
    return resp.json()


# ── Study info ───────────────────────────────────────────────────────────────

def get_study_info(study_id: str) -> dict:
    info = api_get(f"studies/{study_id}")
    print(f"Study: {info['name']}")
    print(f"Description: {info.get('description', 'N/A')[:200]}")
    print(f"Samples: {info.get('allSampleCount', '?')}")
    print()
    return info


def get_molecular_profiles(study_id: str) -> list[dict]:
    profiles = api_get(f"studies/{study_id}/molecular-profiles")
    print(f"Available molecular profiles ({len(profiles)}):")
    for p in profiles:
        print(f"  {p['molecularProfileId']:55s}  {p['molecularAlterationType']:20s}  {p['datatype']}")
    print()
    return profiles


def find_profile_id(profiles: list[dict], alteration_type: str, datatype: str | None = None) -> str | None:
    for p in profiles:
        if p["molecularAlterationType"] == alteration_type:
            if datatype is None or p["datatype"] == datatype:
                return p["molecularProfileId"]
    return None


def get_sample_ids(study_id: str) -> list[str]:
    samples = api_get(f"studies/{study_id}/samples", params={"projection": "ID"})
    return [s["sampleId"] for s in samples]


# ── Data fetchers ────────────────────────────────────────────────────────────

def fetch_mutations(profiles: list[dict], sample_ids: list[str]) -> pd.DataFrame:
    profile_id = find_profile_id(profiles, "MUTATION_EXTENDED")
    if not profile_id:
        print("⚠ No mutation profile found.")
        return pd.DataFrame()
    print(f"Fetching mutations from '{profile_id}' ...")
    payload = {
        "sampleMolecularIdentifiers": [
            {"molecularProfileId": profile_id, "sampleId": sid}
            for sid in sample_ids
        ],
    }
    data = api_post("mutations/fetch?projection=DETAILED", payload)
    df = pd.json_normalize(data)
    print(f"  → {len(df)} mutation records")
    return df


def fetch_fusions(profiles: list[dict], sample_ids: list[str]) -> pd.DataFrame:
    profile_id = find_profile_id(profiles, "STRUCTURAL_VARIANT")
    if profile_id:
        print(f"Fetching fusions from '{profile_id}' ...")
        payload = {
            "sampleMolecularIdentifiers": [
                {"molecularProfileId": profile_id, "sampleId": sid}
                for sid in sample_ids
            ],
        }
        data = api_post("structural-variant/fetch", payload)
        df = pd.json_normalize(data)
        print(f"  → {len(df)} fusion/structural variant records")
        return df

    print("No dedicated structural variant profile. Checking mutations for fusion events...")
    mut_profile = find_profile_id(profiles, "MUTATION_EXTENDED")
    if mut_profile:
        payload = {
            "sampleMolecularIdentifiers": [
                {"molecularProfileId": mut_profile, "sampleId": sid}
                for sid in sample_ids
            ],
        }
        data = api_post("mutations/fetch?projection=DETAILED", payload)
        fusions = [r for r in data if r.get("mutationType", "").upper() == "FUSION"]
        df = pd.json_normalize(fusions)
        print(f"  → {len(df)} fusion records extracted from mutation data")
        return df

    print("⚠ No fusion data found.")
    return pd.DataFrame()


def fetch_cnv(profiles: list[dict], sample_ids: list[str]) -> pd.DataFrame:
    profile_id = find_profile_id(profiles, "COPY_NUMBER_ALTERATION", datatype="DISCRETE")
    if not profile_id:
        print("⚠ No discrete CNA profile found.")
        return pd.DataFrame()
    print(f"Fetching discrete CNA from '{profile_id}' ...")
    payload = {"sampleIds": sample_ids}
    data = api_post(
        f"molecular-profiles/{profile_id}/discrete-copy-number/fetch?"
        "discreteCopyNumberEventType=ALL&projection=DETAILED",
        payload,
    )
    df = pd.json_normalize(data)
    print(f"  → {len(df)} CNA records")
    return df


# ── Clinical data ────────────────────────────────────────────────────────────

def fetch_clinical(study_id: str, data_type: str) -> pd.DataFrame:
    data = api_get(
        f"studies/{study_id}/clinical-data",
        params={"clinicalDataType": data_type, "projection": "DETAILED"},
    )
    if not data:
        return pd.DataFrame()
    df_long = pd.json_normalize(data)
    if data_type == "PATIENT":
        df_wide = df_long.pivot_table(
            index="patientId", columns="clinicalAttributeId",
            values="value", aggfunc="first",
        ).reset_index()
    else:
        df_wide = df_long.pivot_table(
            index=["patientId", "sampleId"], columns="clinicalAttributeId",
            values="value", aggfunc="first",
        ).reset_index()
    df_wide.columns.name = None
    return df_wide


def fetch_all_clinical(study_id: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("Fetching clinical data ...")
    patient_df = fetch_clinical(study_id, "PATIENT")
    sample_df = fetch_clinical(study_id, "SAMPLE")

    if patient_df.empty and sample_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    if patient_df.empty:
        return pd.DataFrame(), sample_df, sample_df
    if sample_df.empty:
        return patient_df, pd.DataFrame(), patient_df

    merged = sample_df.merge(patient_df, on="patientId", how="outer", suffixes=("", "_patient"))
    print(f"  → {len(patient_df)} patients, {len(sample_df)} samples, "
          f"{merged.shape[1]} total attributes")
    return patient_df, sample_df, merged


# ── README ───────────────────────────────────────────────────────────────────

def write_readme(output_dir, study_id, study_info, profiles, sample_ids,
                 mutations_df, fusions_df, cnv_df, clinical_df):
    lines = []
    w = lines.append

    w(f"# cBioPortal Data Download: {study_info['name']}")
    w("")
    w(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    w("")
    w("## Study")
    w("")
    w(f"- **Study ID:** `{study_id}`")
    w(f"- **Name:** {study_info['name']}")
    desc = study_info.get('description', 'N/A')
    w(f"- **Description:** {desc[:300]}{'...' if len(desc) > 300 else ''}")
    w(f"- **Total samples:** {len(sample_ids)}")
    w(f"- **Source:** https://www.cbioportal.org/study/summary?id={study_id}")
    w("")

    w("## Molecular Profiles")
    w("")
    w("| Profile ID | Alteration Type | Data Type |")
    w("|---|---|---|")
    for p in profiles:
        w(f"| `{p['molecularProfileId']}` | {p['molecularAlterationType']} | {p['datatype']} |")
    w("")

    w("## Downloaded Data")
    w("")
    for label, df in [("Mutations", mutations_df), ("Fusions", fusions_df), ("CNV", cnv_df)]:
        w(f"### {label}")
        w("")
        if df.empty:
            w("Not available.")
        else:
            w(f"- **Records:** {len(df)}")
            if "sampleId" in df.columns:
                w(f"- **Unique samples:** {df['sampleId'].nunique()}")
            gene_col = next((c for c in ["gene.hugoGeneSymbol", "hugoGeneSymbol"] if c in df.columns), None)
            if gene_col:
                w(f"- **Unique genes:** {df[gene_col].nunique()}")
        w("")

    if not clinical_df.empty:
        w("### Clinical Data")
        w("")
        w(f"- **Rows:** {len(clinical_df)}")
        w(f"- **Attributes:** {clinical_df.shape[1]}")
        w(f"- **Columns:** {', '.join(sorted(clinical_df.columns)[:30])}")
    w("")

    (output_dir / "README.md").write_text("\n".join(lines))
    print(f"Saved README    → {output_dir / 'README.md'}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Download data from cBioPortal")
    parser.add_argument("study_id", help="cBioPortal study ID (e.g. laml_tcga_pub, breast_msk_2025)")
    parser.add_argument("--data-dir", help="Base data directory", default="data")
    args = parser.parse_args()

    study_id = args.study_id
    data_dir = Path(args.data_dir)
    output_dir = data_dir / "output" / study_id / "cbioportal_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"cBioPortal Data Fetcher")
    print("=" * 60)
    print(f"Study:  {study_id}")
    print(f"Output: {output_dir}\n")

    study_info = get_study_info(study_id)
    profiles = get_molecular_profiles(study_id)
    sample_ids = get_sample_ids(study_id)
    print(f"Total samples: {len(sample_ids)}\n")

    # Fetch genomic data
    mutations_df = fetch_mutations(profiles, sample_ids)
    fusions_df = fetch_fusions(profiles, sample_ids)
    cnv_df = fetch_cnv(profiles, sample_ids)

    # Fetch clinical data
    patient_df, sample_df, clinical_df = fetch_all_clinical(study_id)

    # Save
    for name, df in [("mutations", mutations_df), ("fusions", fusions_df),
                     ("cnv", cnv_df), ("clinical_outcomes", clinical_df),
                     ("patient_clinical", patient_df)]:
        if not df.empty:
            out = output_dir / f"{name}.csv"
            df.to_csv(out, index=False)
            print(f"Saved {name:20s} → {out}  ({len(df)} rows)")

    write_readme(output_dir, study_id, study_info, profiles, sample_ids,
                 mutations_df, fusions_df, cnv_df, clinical_df)

    print(f"\nDone. Output in: {output_dir.resolve()}")


if __name__ == "__main__":
    main()