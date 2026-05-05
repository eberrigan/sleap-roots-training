"""Clean up orphan SLPs + non-target predictions.csv rows in deployed_2022/predict_output.

These are leftover from a prior un-scoped attempt; the scoped run produced 452
in-target SLPs but the cleanup before re-launch missed ~806 prior SLPs.

Steps:
1. Read the scoped target scan_ids from predict_output/scans.csv (the file we
   wrote via setup_scoped_pipeline_input.py — informational, but the right list).
2. List all *.slp in predict_output/ and identify orphans (scan_id NOT in target).
3. Read predictions.csv; identify non-target rows.
4. Back up predictions.csv -> predictions.csv.bak
5. Write filtered predictions.csv (target rows only).
6. Delete orphan SLPs.

Each step prints what it's doing. Pass --execute to actually delete; default is dry-run.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

PRED_DIR = Path(r"Z:/users/eberrigan/20260401_Javier_Martinez_Pacheco_TTC_SALK_Soybean/2026-05-08_aug_retrain/deployed_2022/predict_output")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="actually delete (default is dry-run)")
    args = parser.parse_args()
    mode = "EXECUTE" if args.execute else "DRY-RUN"
    print(f"=== Mode: {mode} ===\n")

    # 1. Source-of-truth for "what to keep": the scan_ids actually present in
    # traits_summary.csv (these are the 452 the dataset_params.json range-filter
    # produced during the scoped run, INCLUDING wave 4/5 which are filtered out
    # only at the analysis step). The smaller 296-row scans.csv is a more
    # restrictive ideal target but we should keep everything the analysis touched.
    traits_csv = PRED_DIR.parent / "traits_output" / "traits_summary.csv"
    if not traits_csv.exists():
        sys.exit(f"missing {traits_csv}")
    target_ids = set(pd.read_csv(traits_csv)["scan_id"].astype(str).tolist())
    print(f"Target scan_ids (from traits_summary.csv — actual scoped 452): {len(target_ids)}")
    print(f"  e.g.: {sorted(list(target_ids))[:3]}")

    # 2. find orphan SLPs
    all_slps = list(PRED_DIR.glob("scan_*.slp"))
    print(f"\nTotal SLP files: {len(all_slps)}")

    def slp_scan_id(p: Path) -> str:
        # Filename: scan_<id>.model_<...>.root_{primary,lateral}.slp
        # extract <id>
        return p.name.split(".", 1)[0].replace("scan_", "")

    target_slps = [p for p in all_slps if slp_scan_id(p) in target_ids]
    orphan_slps = [p for p in all_slps if slp_scan_id(p) not in target_ids]
    print(f"  in-target: {len(target_slps)}")
    print(f"  orphans (not in target): {len(orphan_slps)}")
    if orphan_slps[:3]:
        print(f"  first 3 orphans: {[p.name for p in orphan_slps[:3]]}")

    # 3. predictions.csv stats
    preds_csv = PRED_DIR / "predictions.csv"
    preds_df = pd.read_csv(preds_csv)
    print(f"\npredictions.csv: {len(preds_df)} rows")
    preds_df["scan_id_str"] = preds_df["scan_id"].astype(str)
    target_mask = preds_df["scan_id_str"].isin(target_ids)
    print(f"  in-target rows: {target_mask.sum()}")
    print(f"  orphan rows: {(~target_mask).sum()}")

    # 4-6. Apply changes if --execute
    if not args.execute:
        print("\nDRY-RUN — no changes made. Re-run with --execute to apply.")
        return

    # 4. backup predictions.csv
    bak = preds_csv.with_suffix(".csv.bak")
    if not bak.exists():
        shutil.copy2(preds_csv, bak)
        print(f"\nBacked up {preds_csv.name} -> {bak.name}")
    else:
        print(f"\nBackup already exists at {bak.name} (not overwriting)")

    # 5. write filtered predictions.csv (drop helper col)
    filtered = preds_df[target_mask].drop(columns=["scan_id_str"])
    filtered.to_csv(preds_csv, index=False)
    print(f"  wrote filtered predictions.csv with {len(filtered)} rows (was {len(preds_df)})")

    # 6. delete orphan SLPs
    for p in orphan_slps:
        p.unlink()
    print(f"  deleted {len(orphan_slps)} orphan SLP files")

    # final state
    remaining = list(PRED_DIR.glob("scan_*.slp"))
    print(f"\nRemaining SLPs: {len(remaining)} (expect ~{len(target_slps)} = {len(target_slps)//2} primary + {len(target_slps)//2} lateral)")


if __name__ == "__main__":
    main()
