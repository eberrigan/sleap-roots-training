"""Extract accession_id -> genotype mapping from the original Apr-2026 Javier
pipeline output, since that file already has the enrichment done."""
import pandas as pd

src = r'Z:/users/eberrigan/20260401_Javier_Martinez_Pacheco_TTC_SALK_Soybean/sleap_roots_traits_output/traits_summary_with_genotype.csv'
df = pd.read_csv(src)
print(f'Total rows: {len(df)}')
print(f'Unique accession_ids: {df["accession_id"].nunique()}')
print(f'Unique genotypes: {df["genotype"].nunique()}')
print()
print('Sample mapping (first 10 unique pairs):')
mapping = df[['accession_id', 'genotype']].drop_duplicates().sort_values('accession_id')
print(mapping.head(10).to_string(index=False))
print()
print(f'Null genotypes: {df["genotype"].isna().sum()}')
print(f'Null accession_ids: {df["accession_id"].isna().sum()}')
print()
# Save mapping for reuse on retrain conditions
mapping.to_csv(r'Z:/users/eberrigan/20260401_Javier_Martinez_Pacheco_TTC_SALK_Soybean/2026-05-08_aug_retrain/analysis/accession_to_genotype.csv', index=False)
print(f'Saved mapping ({len(mapping)} rows) -> analysis/accession_to_genotype.csv')
