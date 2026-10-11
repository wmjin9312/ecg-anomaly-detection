
from pathlib import Path
from collections import Counter

import numpy as np
import wfdb
from sklearn.model_selection import GroupShuffleSplit

DATA_DIR = Path("data/raw/mitdb")
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
BEFORE = 90
AFTER = 144
SEQ_LEN = BEFORE + AFTER

# AAMI 기반 심박 그룹
NORMAL = {"N", "L", "R", "e", "j"}
SUPRAVENTRICULAR = {"A", "a", "J", "S"}
VENTRICULAR = {"V", "E"}
FUSION = {"F"}

ABNORMAL = SUPRAVENTRICULAR | VENTRICULAR | FUSION
VALID_SYMBOLS = NORMAL | ABNORMAL


def patient_group(record_id):
    # 201과 202는 동일한 환자
    return "201_202" if record_id in {"201", "202"} else record_id


def extract_record(record_id):
    path = str(DATA_DIR / record_id)

    record = wfdb.rdrecord(path)
    ann = wfdb.rdann(path, "atr")

    # 첫 번째 ECG 채널 사용
    signal = record.p_signal[:, 0]

    X, y, groups, symbols = [], [], [], []

    for pos, symbol in zip(ann.sample, ann.symbol):
        if symbol not in VALID_SYMBOLS:
            continue

        start = pos - BEFORE
        end = pos + AFTER

        if start < 0 or end > len(signal):
            continue

        beat = signal[start:end].astype(np.float32)

        if len(beat) != SEQ_LEN:
            continue

        if not np.isfinite(beat).all():
            continue

        # 각 박동별 표준화
        mean = beat.mean()
        std = beat.std()

        if std < 1e-6:
            continue

        beat = (beat - mean) / std

        label = 0 if symbol in NORMAL else 1

        X.append(beat)
        y.append(label)
        groups.append(patient_group(record_id))
        symbols.append(symbol)

    return X, y, groups, symbols


def save_split(name, X, y, groups, symbols):
    np.savez_compressed(
        OUTPUT_DIR / f"{name}.npz",
        X=X,
        y=y,
        groups=groups,
        symbols=symbols,
    )

    counts = Counter(y.tolist())

    print(
        f"{name:12s} | "
        f"Samples: {len(y):6d} | "
        f"Normal: {counts[0]:6d} | "
        f"Abnormal: {counts[1]:6d} | "
        f"Patients: {len(set(groups))}"
    )


def main():
    records = sorted(p.stem for p in DATA_DIR.glob("*.dat"))

    if len(records) != 48:
        raise RuntimeError(
            f"Expected 48 records, found {len(records)}"
        )

    all_X, all_y = [], []
    all_groups, all_symbols = [], []

    for record_id in records:
        X, y, groups, symbols = extract_record(record_id)

        all_X.extend(X)
        all_y.extend(y)
        all_groups.extend(groups)
        all_symbols.extend(symbols)

        print(f"[OK] Record {record_id}: {len(X)} beats")

    X = np.asarray(all_X, dtype=np.float32)
    y = np.asarray(all_y, dtype=np.int64)
    groups = np.asarray(all_groups)
    symbols = np.asarray(all_symbols)

    # 환자 단위 Train / 나머지 분리
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.30,
        random_state=SEED,
    )

    train_idx, temp_idx = next(
        splitter.split(X, y, groups)
    )

    # 나머지 환자를 Validation / Test로 분리
    splitter2 = GroupShuffleSplit(
        n_splits=1,
        test_size=0.50,
        random_state=SEED,
    )

    val_local, test_local = next(
        splitter2.split(
            X[temp_idx],
            y[temp_idx],
            groups[temp_idx],
        )
    )

    val_idx = temp_idx[val_local]
    test_idx = temp_idx[test_local]

    # 환자 그룹 누수 검사
    train_groups = set(groups[train_idx])
    val_groups = set(groups[val_idx])
    test_groups = set(groups[test_idx])

    assert train_groups.isdisjoint(val_groups)
    assert train_groups.isdisjoint(test_groups)
    assert val_groups.isdisjoint(test_groups)

    print("\n========== DATASET SUMMARY ==========")

    # 학습은 정상 박동만 사용
    train_normal_idx = train_idx[y[train_idx] == 0]

    for name, idx in [
        ("train_normal", train_normal_idx),
        ("validation", val_idx),
        ("test", test_idx),
    ]:
        save_split(
            name,
            X[idx],
            y[idx],
            groups[idx],
            symbols[idx],
        )

    print("\nPatient-level leakage check: PASS")
    print("Sequence length:", SEQ_LEN)


if __name__ == "__main__":
    main()
