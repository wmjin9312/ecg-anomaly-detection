
from pathlib import Path
import json

import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

DATA_DIR = Path("data/processed")
RESULT_DIR = Path("results/isolation_forest")
RESULT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42


def load_data(filename):
    with np.load(DATA_DIR / filename) as data:
        X = data["X"].astype(np.float32)
        y = data["y"].astype(np.int64)
        groups = data["groups"].copy()

    return X, y, groups


def select_threshold(y_true, scores):
    precision, recall, thresholds = precision_recall_curve(
        y_true, scores
    )

    # 마지막 precision/recall 값에는 대응하는 threshold가 없음
    f1 = (
        2 * precision[:-1] * recall[:-1]
        / (precision[:-1] + recall[:-1] + 1e-12)
    )

    best_idx = int(np.argmax(f1))

    return float(thresholds[best_idx]), float(f1[best_idx])


def main():
    # 1. 데이터 로딩
    X_train, y_train, g_train = load_data("train_normal.npz")
    X_val, y_val, g_val = load_data("validation.npz")
    X_test, y_test, g_test = load_data("test.npz")

    assert np.all(y_train == 0)
    assert set(g_train).isdisjoint(set(g_val))
    assert set(g_train).isdisjoint(set(g_test))
    assert set(g_val).isdisjoint(set(g_test))
    assert len(np.unique(y_val)) == 2
    assert len(np.unique(y_test)) == 2

    print("=== Dataset ===")
    print("Train:", X_train.shape)
    print("Validation:", X_val.shape)
    print("Test:", X_test.shape)

    # 2. Isolation Forest 학습
    print("\nTraining Isolation Forest...")

    model = IsolationForest(
        n_estimators=200,
        max_samples=256,
        contamination="auto",
        random_state=SEED,
        n_jobs=-1,
    )

    model.fit(X_train)

    # 3. 이상 점수 계산
    # score_samples는 작을수록 비정상이므로 부호 반전
    val_scores = -model.score_samples(X_val)
    test_scores = -model.score_samples(X_test)

    # 4. Validation에서 Threshold 선정
    threshold, val_f1 = select_threshold(
        y_val, val_scores
    )

    print(f"\nSelected Threshold: {threshold:.6f}")
    print(f"Validation F1: {val_f1:.4f}")

    # 5. Test 평가
    # precision_recall_curve의 threshold는 >= 기준
    y_pred = (test_scores >= threshold).astype(int)

    metrics = {
        "threshold": threshold,
        "validation_f1": val_f1,
        "test_precision": float(
            precision_score(y_test, y_pred, zero_division=0)
        ),
        "test_recall": float(
            recall_score(y_test, y_pred, zero_division=0)
        ),
        "test_f1": float(
            f1_score(y_test, y_pred, zero_division=0)
        ),
        "test_pr_auc": float(
            average_precision_score(y_test, test_scores)
        ),
        "test_roc_auc": float(
            roc_auc_score(y_test, test_scores)
        ),
    }

    print("\n========== TEST RESULTS ==========")

    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    # 6. Confusion Matrix
    cm = confusion_matrix(
        y_test, y_pred, labels=[0, 1]
    )

    fig, ax = plt.subplots(figsize=(6, 5))

    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Normal", "Abnormal"],
    ).plot(ax=ax, cmap="Blues", colorbar=False)

    ax.set_title("Isolation Forest - Confusion Matrix")
    fig.tight_layout()

    fig.savefig(
        RESULT_DIR / "confusion_matrix.png",
        dpi=200
    )
    plt.close(fig)

    # 7. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, test_scores)

    fig, ax = plt.subplots(figsize=(6, 5))

    ax.plot(
        fpr, tpr,
        label=f"ROC-AUC = {metrics['test_roc_auc']:.4f}"
    )
    ax.plot([0, 1], [0, 1], "--", color="gray")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Isolation Forest - ROC Curve")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(RESULT_DIR / "roc_curve.png", dpi=200)
    plt.close(fig)

    # 8. Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(
        y_test, test_scores
    )

    fig, ax = plt.subplots(figsize=(6, 5))

    ax.plot(
        recall, precision,
        label=f"AP = {metrics['test_pr_auc']:.4f}"
    )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Isolation Forest - PR Curve")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(RESULT_DIR / "pr_curve.png", dpi=200)
    plt.close(fig)

    # 9. 결과 저장
    with open(
        RESULT_DIR / "metrics.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(metrics, f, indent=4)

    np.savez_compressed(
        RESULT_DIR / "test_predictions.npz",
        y_true=y_test,
        y_pred=y_pred,
        scores=test_scores,
    )

    print("\nConfusion Matrix:")
    print(cm)

    print(f"\nResults saved to: {RESULT_DIR.resolve()}")


if __name__ == "__main__":
    main()
