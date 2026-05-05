"""Step 1 of /configure-run-all: dataset inspection + statistical guardrails."""
from pathlib import Path
import sys
sys.path.insert(0, r"C:/repos/sleap-roots-analyze/src")

from sleap_roots_analyze.config_authoring import (
    inspect_dataset,
    warn_mahalanobis_small_n,
    warn_heritability_low_replicates,
    recommend_umap_n_neighbors,
)
import pandas as pd

CSV = r"Z:/users/eberrigan/20260401_Javier_Martinez_Pacheco_TTC_SALK_Soybean/2026-05-08_aug_retrain/brightness/traits_output/traits_summary_with_genotype.csv"

print(f"=== Dataset inspection: brightness retrain traits ===\n")
result = inspect_dataset(CSV)
print(f"n_samples: {result['n_samples']}")
print(f"n_numeric_cols: {result['n_numeric_cols']}")
print()
print(f"Group-by candidates (cols with <=20 unique values):")
print(f"  {result['group_by_candidates']}")
print()

# Check group sizes vs Mahalanobis threshold (n >= 30)
print("=== Mahalanobis guardrail (n >= 30 per group) ===")
df = pd.read_csv(CSV)
print(f"plant_age_days groups: {df['plant_age_days'].value_counts().to_dict()}")
for v, n in df['plant_age_days'].value_counts().items():
    msg = warn_mahalanobis_small_n(n)
    if msg:
        print(f"  age={v}: n={n} -> {msg}")
    else:
        print(f"  age={v}: n={n} -> OK for Mahalanobis")
print()

# Heritability: min replicates per genotype
print("=== Heritability guardrail (>= 3 reps/genotype) ===")
min_reps = df.groupby("genotype")["scan_id"].nunique().min()
mean_reps = df.groupby("genotype")["scan_id"].nunique().mean()
n_genotypes = df["genotype"].nunique()
print(f"n_genotypes: {n_genotypes}")
print(f"min_reps_per_genotype: {min_reps}")
print(f"mean_reps_per_genotype: {mean_reps:.2f}")
msg = warn_heritability_low_replicates(min_reps)
if msg:
    print(f"  -> {msg}")
else:
    print(f"  -> OK for heritability")
print()

# UMAP n_neighbors recommendation
print("=== UMAP n_neighbors recommendation ===")
n, warning = recommend_umap_n_neighbors(result['n_samples'])
print(f"n_samples={result['n_samples']} -> recommended n_neighbors={n}")
if warning:
    print(f"  -> {warning}")

print()
print(f"Wave distribution at Day 10 (the only group, since this is scoped):")
print(df[df['plant_age_days'] == 10]['wave_number'].value_counts().sort_index().to_dict())
