
from collections import Counter
from pathlib import Path

import wfdb

DATA_DIR = Path("data/raw/mitdb")

RECORDS = ["102", "104", "107", "207", "217"]

NORMAL = {"N", "L", "R", "e", "j"}
ABNORMAL = {"A", "a", "J", "S", "V", "E", "F"}

for record_id in RECORDS:
    path = str(DATA_DIR / record_id)

    record = wfdb.rdrecord(path)
    ann = wfdb.rdann(path, "atr")

    counts = Counter(ann.symbol)

    print(f"\n{'=' * 48}")
    print(f"Record: {record_id}")
    print(f"Channels: {record.sig_name}")

    print("\nAnnotation distribution:")
    for symbol, count in counts.most_common():
        print(f"  {symbol}: {count}")

    print("\nNormal group:", sum(
        counts[s] for s in NORMAL
    ))

    print("Abnormal group:", sum(
        counts[s] for s in ABNORMAL
    ))

    print("Excluded annotations:", sum(
        count
        for symbol, count in counts.items()
        if symbol not in NORMAL | ABNORMAL
    ))


import numpy as np
import pandas as pd

with np.load("data/processed/test.npz") as data:
    y = data["y"]
    groups = data["groups"].astype(str)
    symbols = data["symbols"].astype(str)

models = {
    "Isolation Forest": "results/isolation_forest/test_predictions.npz",
    "LSTM Autoencoder": "results/lstm_autoencoder/test_predictions.npz",
}

mask_207 = groups == "207"

print("\n========== RECORD 207 ERROR ANALYSIS ==========")

for model_name, path in models.items():
    with np.load(path) as data:
        y_true = data["y_true"]
        y_pred = data["y_pred"]

    assert np.array_equal(y_true, y)

    df = pd.DataFrame({
        "symbol": symbols[mask_207],
        "true": y[mask_207],
        "pred": y_pred[mask_207],
    })

    df["result"] = np.select(
        [
            (df.true == 0) & (df.pred == 1),
            (df.true == 1) & (df.pred == 0),
            (df.true == 1) & (df.pred == 1),
        ],
        ["FP", "FN", "TP"],
        default="TN",
    )

    print(f"\n[{model_name}]")
    print(pd.crosstab(df["symbol"], df["result"]))



from collections import Counter
import numpy as np

print("\n========== NORMAL CLASS DISTRIBUTION ==========")

for split in ["train_normal", "validation", "test"]:
    with np.load(f"data/processed/{split}.npz") as data:
        symbols = data["symbols"].astype(str)
        groups = data["groups"].astype(str)

    counts = Counter(symbols)

    print(f"\n[{split}]")
    print("Total:", len(symbols))

    for symbol in ["N", "L", "R", "e", "j"]:
        count = counts[symbol]
        ratio = count / len(symbols) * 100
        print(f"{symbol}: {count} ({ratio:.2f}%)")

    if split == "train_normal":
        for symbol in ["L", "R"]:
            mask = symbols == symbol
            patient_counts = Counter(groups[mask])
            print(
                f"\n{symbol} by patient:",
                patient_counts.most_common()
            )
