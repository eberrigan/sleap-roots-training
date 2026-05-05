"""Step 7 of /configure-run-all: validate the brightness QC + Viz configs."""
import sys
sys.path.insert(0, r"C:/repos/sleap-roots-analyze/src")

from sleap_roots_analyze.pipeline.config.utils import (
    load_qc_config,
    load_viz_config,
    validate_qc_config,
    validate_viz_config,
)

QC = r"C:/repos/sleap-roots-analyze/configs/active/qc/javier_ttc_salk_soybean_brightness.yaml"
VIZ = r"C:/repos/sleap-roots-analyze/configs/active/viz/javier_ttc_salk_soybean_brightness.yaml"

print("=== QC config validation ===")
try:
    qc_config = load_qc_config(QC)
    validate_qc_config(qc_config)
    print(f"  OK: {QC}")
except Exception as e:
    print(f"  FAIL: {type(e).__name__}: {e}")

print()
print("=== Viz config validation ===")
try:
    viz_config = load_viz_config(VIZ)
    validate_viz_config(viz_config)
    print(f"  OK: {VIZ}")
except Exception as e:
    print(f"  FAIL: {type(e).__name__}: {e}")
