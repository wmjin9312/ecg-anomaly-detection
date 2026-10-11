
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

RESULT_DIR = Path("results/l_beat_analysis")
RESULT_DIR.mkdir(parents=True, exist_ok=True)


def load_l_beats(split):
    with np.load(f"data/processed/{split}.npz") as data:
        X = data["X"]
        symbols = data["symbols"].astype(str)
        groups = data["groups"].astype(str)

    mask = symbols == "L"
    return X[mask], groups[mask]


train_X, train_groups = load_l_beats("train_normal")
test_X, test_groups = load_l_beats("test")

plt.figure(figsize=(12, 5))

for record_id in ["109", "111"]:
    X = train_X[train_groups == record_id]

    if len(X):
        plt.plot(
            X.mean(axis=0),
            label=f"Train Record {record_id} (n={len(X)})"
        )

X_207 = test_X[test_groups == "207"]

plt.plot(
    X_207.mean(axis=0),
    label=f"Test Record 207 (n={len(X_207)})",
    linewidth=2,
    linestyle="--"
)

plt.xlabel("Sample")
plt.ylabel("Normalized Amplitude")
plt.title("Mean L-Beat Waveforms by Record")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(
    RESULT_DIR / "mean_l_waveforms.png",
    dpi=200
)

plt.show()

print("Train L beats:", len(train_X))
print("Test Record 207 L beats:", len(X_207))
