"""Count scan distribution by wave at each plant_age_days."""
import pandas as pd
df = pd.read_csv(r"Z:/users/eberrigan/20260401_Javier_Martinez_Pacheco_TTC_SALK_Soybean/images_downloader_output/scans.csv")
print(f"Total scans: {len(df)}")
print(f"Species in csv: {df['species_name'].unique()}")
print()
print("Scans by (plant_age_days, wave_number):")
counts = df.groupby(["plant_age_days", "wave_number"]).size().unstack(fill_value=0)
print(counts)
print()
day10 = df[df["plant_age_days"] == 10]
print(f"\nDay-10 total: {len(day10)}")
print(f"Day-10 by wave_number: {day10['wave_number'].value_counts().sort_index().to_dict()}")
