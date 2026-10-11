
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

DATA_DIR = Path("data/processed")
RESULT_DIR = Path("results/error_analysis")
RESULT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = {
    "Isolation Forest": Path(
        "results/isolation_forest/test_predictions.npz"
    ),
    "LSTM Autoencoder": Path(
        "results/lstm_autoencoder/test_predictions.npz"
    ),
}

# AAMI 기반 이상 박동 유형
BEAT_GROUPS = {
    "S": {"A", "a", "J", "S"},
    "V": {"V", "E"},
    "F": {"F"},
}


def get_beat_group(symbol):
    for group, symbols in BEAT_GROUPS.items():
        if symbol in symbols:
            return group
    return "N"


def load_predictions(path, expected_y):
    with np.load(path) as data:
        y_true = data["y_true"].astype(int)
        y_pred = data["y_pred"].astype(int)
        scores = data["scores"].astype(float)

    assert len(y_true) == len(expected_y)
    assert np.array_equal(y_true, expected_y), (
        f"Label mismatch: {path}"
    )
    assert len(y_pred) == len(scores) == len(expected_y)

    return y_pred, scores


def main():
    # 1. 테스트 데이터
    with np.load(DATA_DIR / "test.npz") as data:
        X = data["X"]
        y = data["y"].astype(int)
        groups = data["groups"].astype(str)
        symbols = data["symbols"].astype(str)

    df = pd.DataFrame({
        "patient": groups,
        "symbol": symbols,
        "beat_group": [
            get_beat_group(s) for s in symbols
        ],
        "y_true": y,
    })

    # 2. 모델별 예측 불러오기
    for model_name, path in MODELS.items():
        pred, scores = load_predictions(path, y)
        prefix = (
            "iforest" if model_name == "Isolation Forest"
            else "lstm"
        )

        df[f"{prefix}_pred"] = pred
        df[f"{prefix}_score"] = scores

    # 3. 환자 그룹별 평가
    patient_results = []

    for patient, group in df.groupby("patient"):
        for model, prefix in [
            ("Isolation Forest", "iforest"),
            ("LSTM Autoencoder", "lstm"),
        ]:
            true = group["y_true"].to_numpy()
            pred = group[f"{prefix}_pred"].to_numpy()

            tn, fp, fn, tp = confusion_matrix(
                true, pred, labels=[0, 1]
            ).ravel()

            patient_results.append({
                "patient": patient,
                "model": model,
                "samples": len(group),
                "abnormal": int(true.sum()),
                "precision": precision_score(
                    true, pred, zero_division=0
                ),
                "recall": recall_score(
                    true, pred, zero_division=0
                ),
                "f1": f1_score(
                    true, pred, zero_division=0
                ),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
                "tn": int(tn),
            })

    patient_df = pd.DataFrame(patient_results)
    patient_df.to_csv(
        RESULT_DIR / "patient_metrics.csv",
        index=False,
    )

    # 4. 이상 유형별 Recall
    type_results = []

    for beat_type in ["S", "V", "F"]:
        subset = df[df["beat_group"] == beat_type]

        for model, prefix in [
            ("Isolation Forest", "iforest"),
            ("LSTM Autoencoder", "lstm"),
        ]:
            total = len(subset)
            detected = int(
                subset[f"{prefix}_pred"].sum()
            )

            type_results.append({
                "beat_type": beat_type,
                "model": model,
                "total": total,
                "detected": detected,
                "recall": (
                    detected / total if total else np.nan
                ),
            })

    type_df = pd.DataFrame(type_results)
    type_df.to_csv(
        RESULT_DIR / "beat_type_metrics.csv",
        index=False,
    )

    # 5. 모델 간 예측 비교
    comparison = pd.DataFrame({
        "case": [
            "Both correct",
            "Only Isolation Forest correct",
            "Only LSTM Autoencoder correct",
            "Both wrong",
        ],
        "count": [
            int(((df.iforest_pred == y) &
                 (df.lstm_pred == y)).sum()),

            int(((df.iforest_pred == y) &
                 (df.lstm_pred != y)).sum()),

            int(((df.iforest_pred != y) &
                 (df.lstm_pred == y)).sum()),

            int(((df.iforest_pred != y) &
                 (df.lstm_pred != y)).sum()),
        ],
    })

    comparison.to_csv(
        RESULT_DIR / "model_comparison.csv",
        index=False,
    )

    # 6. 이상 유형별 Recall 시각화
    pivot = type_df.pivot(
        index="beat_type",
        columns="model",
        values="recall",
    )

    ax = pivot.plot.bar(
        figsize=(9, 5),
        rot=0,
    )

    ax.set_title("Anomaly Recall by Beat Type")
    ax.set_xlabel("AAMI Beat Group")
    ax.set_ylabel("Recall")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        RESULT_DIR / "beat_type_recall.png",
        dpi=200,
    )
    plt.close()

    # 7. 환자별 F1 시각화
    pivot = patient_df.pivot(
        index="patient",
        columns="model",
        values="f1",
    )

    ax = pivot.plot.bar(
        figsize=(13, 5),
        rot=0,
    )

    ax.set_title("F1-score by Patient Group")
    ax.set_xlabel("Patient Group")
    ax.set_ylabel("F1-score")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        RESULT_DIR / "patient_f1.png",
        dpi=200,
    )
    plt.close()

    # 8. 오분류 ECG 파형 시각화
    cases = {
        "False Positive": (
            (df.y_true == 0) &
            (df.iforest_pred == 1)
        ),
        "False Negative": (
            (df.y_true == 1) &
            (df.iforest_pred == 0)
        ),
        "Both Wrong": (
            (df.iforest_pred != df.y_true) &
            (df.lstm_pred != df.y_true)
        ),
    }

    for case_name, mask in cases.items():
        indices = np.flatnonzero(mask.to_numpy())[:5]

        if len(indices) == 0:
            continue

        fig, axes = plt.subplots(
            len(indices), 1,
            figsize=(10, 2.2 * len(indices)),
            squeeze=False,
        )

        for ax, idx in zip(axes[:, 0], indices):
            ax.plot(X[idx])
            ax.set_title(
                f"Patient={groups[idx]} | "
                f"Symbol={symbols[idx]} | "
                f"IF={df.iforest_pred.iloc[idx]} | "
                f"LSTM={df.lstm_pred.iloc[idx]}"
            )
            ax.grid(alpha=0.3)

        plt.tight_layout()

        filename = (
            case_name.lower().replace(" ", "_")
            + ".png"
        )
        plt.savefig(RESULT_DIR / filename, dpi=200)
        plt.close()

    print("\n========== PATIENT METRICS ==========")
    print(patient_df.to_string(index=False))

    print("\n========== BEAT TYPE METRICS ==========")
    print(type_df.to_string(index=False))

    print("\n========== MODEL COMPARISON ==========")
    print(comparison.to_string(index=False))

    print("\nResults saved to:", RESULT_DIR.resolve())


if __name__ == "__main__":
    main()
