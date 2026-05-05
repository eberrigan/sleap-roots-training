"""Enrich each retrain condition's traits_summary.csv with a genotype column
(from the lab's existing accession_id -> genotype mapping), then save as
traits_summary_with_genotype.csv (overwriting the prior dummy copies).

Source mapping: extracted from JAVIER_BASE/sleap_roots_traits_output/traits_summary_with_genotype.csv
(94 unique accession_ids -> 94 unique genotype names).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

JAVIER = Path(r"Z:/users/eberrigan/20260401_Javier_Martinez_Pacheco_TTC_SALK_Soybean")
RETRAIN = JAVIER / "2026-05-08_aug_retrain"
ORIG_ENRICHED = JAVIER / "sleap_roots_traits_output" / "traits_summary_with_genotype.csv"
CONDITIONS = ("none", "brightness", "contrast", "both", "deployed_2022")


def main():
    # Build mapping from the original lab-enriched CSV
    src = pd.read_csv(ORIG_ENRICHED)
    mapping = src[["accession_id", "genotype"]].drop_duplicates().set_index("accession_id")["genotype"].to_dict()
    print(f"Loaded mapping: {len(mapping)} accession_id -> genotype pairs")

    for cond in CONDITIONS:
        traits_csv = RETRAIN / cond / "traits_output" / "traits_summary.csv"
        out_csv = RETRAIN / cond / "traits_output" / "traits_summary_with_genotype.csv"
        if not traits_csv.exists():
            print(f"  ⚠ {cond}: missing {traits_csv}")
            continue

        df = pd.read_csv(traits_csv)
        # Drop any pre-existing 'genotype' column (the dummy copy) so we add a fresh one
        if "genotype" in df.columns:
            df = df.drop(columns=["genotype"])
        df["genotype"] = df["accession_id"].map(mapping)
        n_unmapped = df["genotype"].isna().sum()
        df.to_csv(out_csv, index=False)
        unique_geno = df["genotype"].dropna().nunique()
        print(f"  ✓ {cond}: {len(df)} rows, {unique_geno} unique genotypes, {n_unmapped} unmapped"
              f" -> {out_csv.name}")


if __name__ == "__main__":
    main()
