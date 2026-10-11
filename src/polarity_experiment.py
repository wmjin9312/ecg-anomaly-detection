
from pathlib import Path
import json

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
    confusion_matrix,
)

SEED = 42
OUT = Path("results/polarity_experiment")
OUT.mkdir(parents=True, exist_ok=True)


def load_data(name):
    with np.load(f"data/processed/{name}.npz") as d:
        return (
            d["X"].astype(np.float32),
            d["y"].astype(int),
            d["groups"].astype(str),
            d["symbols"].astype(str),
        )


def align_polarity(X):
    """QRS 중심의 주 편향을 양의 방향으로 정렬."""
    X = X.copy()

    # 기존 beat 길이 234, R-peak 기준 위치 90
    center = 90
    half_width = 35
    window = X[:, center-half_width:center+half_width+1]

    # 절댓값이 최대인 지점의 부호를 사용
    peak_idx = np.abs(window).argmax(axis=1)
    peak = window[np.arange(len(X)), peak_idx]

    sign = np.where(peak < 0, -1.0, 1.0)

    return X * sign[:, None]


def choose_threshold(y, scores):
    p, r, thresholds = precision_recall_curve(y, scores)

    f1 = 2 * p[:-1] * r[:-1] / (
        p[:-1] + r[:-1] + 1e-12
    )

    idx = np.argmax(f1)
    return float(thresholds[idx])


def make_validation_split(X, y, groups):
    # 환자 그룹이 겹치지 않는 Calibration / Evaluation
    # 두 부분에 이상 클래스가 존재하는 분할을 탐색
    for seed in range(SEED, SEED + 100):
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=0.5,
            random_state=seed,
        )
        cal_idx, eval_idx = next(
            splitter.split(X, y, groups)
        )

        if (
            len(np.unique(y[cal_idx])) == 2
            and len(np.unique(y[eval_idx])) == 2
        ):
            return cal_idx, eval_idx

    raise RuntimeError("Valid group split not found")


def evaluate(name, X_train, X_cal, y_cal,
             X_eval, y_eval, groups_eval, symbols_eval):

    model = IsolationForest(
        n_estimators=200,
        max_samples=256,
        contamination="auto",
        random_state=SEED,
        n_jobs=-1,
    )

    model.fit(X_train)

    cal_scores = -model.score_samples(X_cal)
    threshold = choose_threshold(y_cal, cal_scores)

    eval_scores = -model.score_samples(X_eval)
    pred = (eval_scores >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_eval, pred, labels=[0, 1]
    ).ravel()

    l_mask = (symbols_eval == "L")

    result = {
        "method": name,
        "threshold": threshold,
        "precision": precision_score(
            y_eval, pred, zero_division=0
        ),
        "recall": recall_score(
            y_eval, pred, zero_division=0
        ),
        "f1": f1_score(
            y_eval, pred, zero_division=0
        ),
        "ap": average_precision_score(
            y_eval, eval_scores
        ),
        "roc_auc": roc_auc_score(
            y_eval, eval_scores
        ),
        "fp": int(fp),
        "fn": int(fn),
        "l_samples": int(l_mask.sum()),
        "l_fpr": (
            float(pred[l_mask].mean())
            if l_mask.any() else None
        ),
    }

    print(f"\n=== {name} ===")
    for key, value in result.items():
        print(f"{key}: {value}")

    return result


def main():
    X_train, y_train, g_train, _ = load_data(
        "train_normal"
    )
    X_val, y_val, groups, symbols = load_data(
        "validation"
    )

    assert np.all(y_train == 0)
    assert set(g_train).isdisjoint(set(groups))

    cal_idx, eval_idx = make_validation_split(
        X_val, y_val, groups
    )

    assert set(groups[cal_idx]).isdisjoint(
        set(groups[eval_idx])
    )

    print("Calibration groups:",
          sorted(set(groups[cal_idx])))
    print("Evaluation groups:",
          sorted(set(groups[eval_idx])))

    results = []

    for name, transform in [
        ("Baseline", lambda x: x),
        ("Polarity Aligned", align_polarity),
    ]:
        result = evaluate(
            name=name,
            X_train=transform(X_train),
            X_cal=transform(X_val[cal_idx]),
            y_cal=y_val[cal_idx],
            X_eval=transform(X_val[eval_idx]),
            y_eval=y_val[eval_idx],
            groups_eval=groups[eval_idx],
            symbols_eval=symbols[eval_idx],
        )
        results.append(result)

    df = pd.DataFrame(results)

    print("\n========== COMPARISON ==========")
    print(df.to_string(index=False))

    df.to_csv(OUT / "comparison.csv", index=False)

    with open(OUT / "split_info.json", "w") as f:
        json.dump({
            "calibration_groups": sorted(
                set(groups[cal_idx])
            ),
            "evaluation_groups": sorted(
                set(groups[eval_idx])
            ),
        }, f, indent=2)

    print("\nSaved:", OUT.resolve())


if __name__ == "__main__":
    main()
