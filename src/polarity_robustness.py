
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
)

from polarity_experiment import (
    load_data,
    align_polarity,
    choose_threshold,
)

SEED = 42
N_SPLITS = 10
OUT = Path("results/polarity_robustness")
OUT.mkdir(parents=True, exist_ok=True)


def main():
    X_train, _, _, _ = load_data("train_normal")
    X_val, y_val, groups, symbols = load_data("validation")

    # 동일한 Train 데이터에서 두 모델 학습
    models = {}

    for name, transform in [
        ("Baseline", lambda x: x),
        ("Polarity Aligned", align_polarity),
    ]:
        model = IsolationForest(
            n_estimators=200,
            max_samples=256,
            contamination="auto",
            random_state=SEED,
            n_jobs=-1,
        )

        model.fit(transform(X_train))
        models[name] = (
            model,
            -model.score_samples(transform(X_val))
        )

    # 환자 그룹 기준 반복 분할
    splitter = GroupShuffleSplit(
        n_splits=N_SPLITS,
        test_size=0.5,
        random_state=SEED,
    )

    rows = []

    for split_id, (cal_idx, eval_idx) in enumerate(
        splitter.split(X_val, y_val, groups), start=1
    ):
        if (
            len(np.unique(y_val[cal_idx])) < 2
            or len(np.unique(y_val[eval_idx])) < 2
        ):
            continue

        for name, (_, scores) in models.items():
            threshold = choose_threshold(
                y_val[cal_idx],
                scores[cal_idx],
            )

            eval_scores = scores[eval_idx]
            eval_y = y_val[eval_idx]
            pred = (eval_scores >= threshold).astype(int)

            l_mask = symbols[eval_idx] == "L"

            rows.append({
                "split": split_id,
                "method": name,
                "precision": precision_score(
                    eval_y, pred, zero_division=0
                ),
                "recall": recall_score(
                    eval_y, pred, zero_division=0
                ),
                "f1": f1_score(
                    eval_y, pred, zero_division=0
                ),
                "ap": average_precision_score(
                    eval_y, eval_scores
                ),
                "roc_auc": roc_auc_score(
                    eval_y, eval_scores
                ),
                "l_samples": int(l_mask.sum()),
                "l_fpr": (
                    float(pred[l_mask].mean())
                    if l_mask.any() else np.nan
                ),
            })

    df = pd.DataFrame(rows)

    df.to_csv(OUT / "split_results.csv", index=False)

    summary = df.groupby("method")[
        ["precision", "recall", "f1", "ap", "roc_auc", "l_fpr"]
    ].agg(["mean", "std"])

    print("\n========== ROBUSTNESS SUMMARY ==========")
    print(summary.to_string())

    print("\n========== L-BEAT SPLITS ==========")
    print(
        df[df["l_samples"] > 0][
            ["split", "method", "l_samples", "l_fpr", "f1"]
        ].to_string(index=False)
    )

    print("\nSaved:", OUT.resolve())


if __name__ == "__main__":
    main()
