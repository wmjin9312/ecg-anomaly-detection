
from pathlib import Path
import wfdb

DATA_DIR = Path("data/raw/mitdb")

records = sorted(p.stem for p in DATA_DIR.glob("*.dat"))

print(f"Found {len(records)} ECG records\n")

success = []
failed = []

for record_id in records:
    path = str(DATA_DIR / record_id)

    try:
        record = wfdb.rdrecord(path)
        ann = wfdb.rdann(path, "atr")

        assert record.p_signal is not None
        assert record.p_signal.shape[0] > 0
        assert len(ann.sample) > 0

        success.append(record_id)
        print(
            f"[OK] {record_id} | "
            f"shape={record.p_signal.shape} | "
            f"annotations={len(ann.sample)}"
        )

    except Exception as e:
        failed.append((record_id, str(e)))
        print(f"[FAIL] {record_id}: {e}")

print("\n========== SUMMARY ==========")
print(f"Successful: {len(success)}")
print(f"Failed: {len(failed)}")

if failed:
    for record_id, error in failed:
        print(record_id, error)
