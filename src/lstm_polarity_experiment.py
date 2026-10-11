
from pathlib import Path
import copy
import json
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import (
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
)


# ============================================================
# 1. Configuration
# ============================================================

SEED = 42

BATCH_SIZE = 128
HIDDEN_SIZE = 64
LATENT_SIZE = 32

LR = 1e-3
EPOCHS = 50
MIN_EPOCHS = 20
PATIENCE = 10

GRAD_CLIP = 1.0

DATA_DIR = Path("data/processed")
RESULT_DIR = Path("results/lstm_polarity_experiment")

RESULT_DIR.mkdir(parents=True, exist_ok=True)

# Same groups as Isolation Forest polarity experiment
CAL_GROUPS = {"203", "214", "231"}
EVAL_GROUPS = {"108", "113", "201_202", "222"}

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# 2. Reproducibility
# ============================================================

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# ============================================================
# 3. Data Loading
# ============================================================

def load_data(name):
    path = DATA_DIR / f"{name}.npz"

    with np.load(path) as data:
        X = data["X"].astype(np.float32)
        y = data["y"].astype(np.int64)
        groups = data["groups"].astype(str)
        symbols = data["symbols"].astype(str)

    return X, y, groups, symbols


# ============================================================
# 4. Polarity Alignment
# ============================================================

def align_polarity(X):
    """
    Align dominant QRS polarity toward positive direction.

    Input:
        X: (N, 234)

    Output:
        float32 array, same shape
    """

    X = np.asarray(X, dtype=np.float32).copy()

    center = 90
    half_width = 35

    window = X[
        :,
        center - half_width:center + half_width + 1
    ]

    peak_idx = np.abs(window).argmax(axis=1)

    peak = window[
        np.arange(len(X)),
        peak_idx
    ]

    sign = np.where(
        peak < 0,
        -1.0,
        1.0
    ).astype(np.float32)

    X = X * sign[:, None]

    return X.astype(np.float32, copy=False)


# ============================================================
# 5. DataLoader
# ============================================================

def make_loader(X, shuffle=False):
    X = np.asarray(X, dtype=np.float32)

    X = np.ascontiguousarray(
        X[:, :, None]
    )

    tensor = torch.from_numpy(X)

    dataset = TensorDataset(tensor)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
    )

    return loader


# ============================================================
# 6. LSTM Autoencoder
# ============================================================

class LSTMAutoencoder(nn.Module):

    def __init__(
        self,
        hidden_size=HIDDEN_SIZE,
        latent_size=LATENT_SIZE
    ):
        super().__init__()

        self.encoder = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            batch_first=True
        )

        self.to_latent = nn.Linear(
            hidden_size,
            latent_size
        )

        self.decoder = nn.LSTM(
            input_size=latent_size,
            hidden_size=hidden_size,
            batch_first=True
        )

        self.output_layer = nn.Linear(
            hidden_size,
            1
        )

    def forward(self, x):

        _, (hidden, _) = self.encoder(x)

        z = self.to_latent(hidden[-1])

        decoder_input = z.unsqueeze(1).expand(
            -1,
            x.size(1),
            -1
        )

        decoded, _ = self.decoder(decoder_input)

        reconstruction = self.output_layer(decoded)

        return reconstruction


# ============================================================
# 7. Train / Validation Epoch
# ============================================================

def run_epoch(model, loader, optimizer=None):

    training = optimizer is not None

    model.train(training)

    total_loss = 0.0
    total_samples = 0

    context = (
        torch.enable_grad()
        if training
        else torch.no_grad()
    )

    with context:

        for (batch,) in loader:

            batch = batch.to(
                DEVICE,
                dtype=torch.float32,
                non_blocking=True
            )

            if training:
                optimizer.zero_grad(set_to_none=True)

            reconstructed = model(batch)

            loss = torch.mean(
                (reconstructed - batch) ** 2
            )

            if not torch.isfinite(loss):
                raise RuntimeError(
                    "Non-finite loss detected"
                )

            if training:

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=GRAD_CLIP
                )

                optimizer.step()

            batch_size = batch.size(0)

            total_loss += loss.item() * batch_size
            total_samples += batch_size

    if total_samples == 0:
        raise RuntimeError("Empty DataLoader")

    return total_loss / total_samples


# ============================================================
# 8. Reconstruction Error
# ============================================================

@torch.no_grad()
def get_anomaly_scores(model, X):

    model.eval()

    loader = make_loader(X, shuffle=False)

    all_scores = []

    for (batch,) in loader:

        batch = batch.to(
            DEVICE,
            dtype=torch.float32,
            non_blocking=True
        )

        reconstructed = model(batch)

        error = torch.mean(
            (batch - reconstructed) ** 2,
            dim=(1, 2)
        )

        all_scores.append(
            error.cpu().numpy()
        )

    return np.concatenate(all_scores)


# ============================================================
# 9. Threshold Selection
# ============================================================

def choose_threshold(y_true, scores):

    precision, recall, thresholds = (
        precision_recall_curve(
            y_true,
            scores
        )
    )

    f1 = (
        2 * precision[:-1] * recall[:-1]
        /
        (
            precision[:-1]
            + recall[:-1]
            + 1e-12
        )
    )

    best_idx = int(np.argmax(f1))

    threshold = float(
        thresholds[best_idx]
    )

    best_f1 = float(
        f1[best_idx]
    )

    return threshold, best_f1


# ============================================================
# 10. Plot Training Curves
# ============================================================

def plot_training(history, name):

    fig, ax = plt.subplots(figsize=(9, 5))

    epochs = range(
        1,
        len(history["train_loss"]) + 1
    )

    ax.plot(
        epochs,
        history["train_loss"],
        label="Train Loss"
    )

    ax.plot(
        epochs,
        history["cal_normal_loss"],
        label="Calibration Normal Loss"
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")

    ax.set_title(
        f"{name} - Training Curve"
    )

    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()

    filename = (
        name.lower().replace(" ", "_")
        + "_training.png"
    )

    fig.savefig(
        RESULT_DIR / filename,
        dpi=200
    )

    plt.close(fig)


# ============================================================
# 11. Plot Confusion Matrix
# ============================================================

def plot_confusion_matrix(y_true, y_pred, name):

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )

    fig, ax = plt.subplots(figsize=(6, 5))

    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Normal", "Abnormal"]
    ).plot(
        ax=ax,
        cmap="Blues",
        colorbar=False
    )

    ax.set_title(
        f"{name} - Confusion Matrix"
    )

    fig.tight_layout()

    filename = (
        name.lower().replace(" ", "_")
        + "_confusion_matrix.png"
    )

    fig.savefig(
        RESULT_DIR / filename,
        dpi=200
    )

    plt.close(fig)

    return cm


# ============================================================
# 12. Run One Experiment
# ============================================================

def run_experiment(
    name,
    X_train,
    X_cal,
    y_cal,
    X_eval,
    y_eval,
    eval_symbols
):

    print("\n" + "=" * 60)
    print("Experiment:", name)
    print("=" * 60)

    # Same initialization seed for both experiments
    set_seed(SEED)

    train_loader = make_loader(
        X_train,
        shuffle=True
    )

    # Normal calibration beats only
    X_cal_normal = X_cal[y_cal == 0]

    if len(X_cal_normal) == 0:
        raise RuntimeError(
            "No normal calibration samples"
        )

    cal_loader = make_loader(
        X_cal_normal,
        shuffle=False
    )

    model = LSTMAutoencoder().to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR
    )

    best_loss = float("inf")
    best_state = None
    best_epoch = 0

    patience_count = 0

    history = {
        "train_loss": [],
        "cal_normal_loss": []
    }

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    for epoch in range(1, EPOCHS + 1):

        train_loss = run_epoch(
            model,
            train_loader,
            optimizer
        )

        cal_loss = run_epoch(
            model,
            cal_loader,
            optimizer=None
        )

        history["train_loss"].append(
            train_loss
        )

        history["cal_normal_loss"].append(
            cal_loss
        )

        print(
            f"[{name}] "
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train: {train_loss:.6f} | "
            f"Cal Normal: {cal_loss:.6f}",
            flush=True
        )

        # Save best model across all epochs
        if cal_loss < best_loss - 1e-5:

            best_loss = cal_loss

            best_state = copy.deepcopy(
                model.state_dict()
            )

            best_epoch = epoch

            if epoch >= MIN_EPOCHS:
                patience_count = 0

        else:
            # Do not accumulate patience before MIN_EPOCHS
            if epoch >= MIN_EPOCHS:
                patience_count += 1

        if (
            epoch >= MIN_EPOCHS
            and patience_count >= PATIENCE
        ):
            print(
                f"Early stopping at epoch {epoch}"
            )
            break

    if best_state is None:
        raise RuntimeError(
            "No valid model checkpoint"
        )

    model.load_state_dict(best_state)

    print(
        f"\nBest Epoch: {best_epoch}"
    )

    print(
        f"Best Calibration Normal Loss: "
        f"{best_loss:.6f}"
    )

    safe_name = (
        name.lower().replace(" ", "_")
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    torch.save(
        {
            "model_state_dict": best_state,
            "hidden_size": HIDDEN_SIZE,
            "latent_size": LATENT_SIZE,
            "best_epoch": best_epoch,
            "best_cal_loss": best_loss
        },
        RESULT_DIR / f"{safe_name}.pt"
    )

    # --------------------------------------------------------
    # Threshold from calibration
    # --------------------------------------------------------

    cal_scores = get_anomaly_scores(
        model,
        X_cal
    )

    threshold, cal_f1 = choose_threshold(
        y_cal,
        cal_scores
    )

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    eval_scores = get_anomaly_scores(
        model,
        X_eval
    )

    y_pred = (
        eval_scores >= threshold
    ).astype(np.int64)

    tn, fp, fn, tp = confusion_matrix(
        y_eval,
        y_pred,
        labels=[0, 1]
    ).ravel()

    l_mask = eval_symbols == "L"

    result = {
        "method": name,
        "best_epoch": int(best_epoch),
        "best_cal_normal_loss": float(best_loss),
        "threshold": float(threshold),
        "calibration_f1": float(cal_f1),
        "precision": float(
            precision_score(
                y_eval,
                y_pred,
                zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                y_eval,
                y_pred,
                zero_division=0
            )
        ),
        "f1": float(
            f1_score(
                y_eval,
                y_pred,
                zero_division=0
            )
        ),
        "ap": float(
            average_precision_score(
                y_eval,
                eval_scores
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                y_eval,
                eval_scores
            )
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "l_samples": int(l_mask.sum()),
        "l_fpr": (
            float(y_pred[l_mask].mean())
            if l_mask.any()
            else None
        )
    }

    # --------------------------------------------------------
    # Save training history
    # --------------------------------------------------------

    pd.DataFrame(history).to_csv(
        RESULT_DIR / f"{safe_name}_history.csv",
        index=False
    )

    plot_training(
        history,
        name
    )

    plot_confusion_matrix(
        y_eval,
        y_pred,
        name
    )

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    np.savez_compressed(
        RESULT_DIR / f"{safe_name}_predictions.npz",
        y_true=y_eval,
        y_pred=y_pred,
        scores=eval_scores
    )

    print(f"\n{name} Evaluation Results:")

    for key, value in result.items():
        print(f"{key}: {value}")

    del model
    del optimizer

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    return result


# ============================================================
# 13. Main
# ============================================================

def main():

    set_seed(SEED)

    print("Device:", DEVICE)

    if DEVICE.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    X_train, y_train, g_train, _ = load_data(
        "train_normal"
    )

    X_val, y_val, groups, symbols = load_data(
        "validation"
    )

    assert np.all(y_train == 0)

    # --------------------------------------------------------
    # Fixed patient-group split
    # --------------------------------------------------------

    cal_mask = np.isin(
        groups,
        list(CAL_GROUPS)
    )

    eval_mask = np.isin(
        groups,
        list(EVAL_GROUPS)
    )

    actual_groups = set(groups)

    expected_groups = CAL_GROUPS | EVAL_GROUPS

    assert actual_groups == expected_groups

    assert not np.any(
        cal_mask & eval_mask
    )

    assert np.all(
        cal_mask | eval_mask
    )

    assert set(g_train).isdisjoint(
        expected_groups
    )

    X_cal = X_val[cal_mask]
    y_cal = y_val[cal_mask]

    X_eval = X_val[eval_mask]
    y_eval = y_val[eval_mask]

    eval_symbols = symbols[eval_mask]

    assert set(np.unique(y_cal)) == {0, 1}
    assert set(np.unique(y_eval)) == {0, 1}

    print("\nDataset:")
    print("Train:", X_train.shape)
    print("Calibration:", X_cal.shape)
    print("Evaluation:", X_eval.shape)

    print(
        "Calibration groups:",
        sorted(CAL_GROUPS)
    )

    print(
        "Evaluation groups:",
        sorted(EVAL_GROUPS)
    )

    # --------------------------------------------------------
    # Input transformation
    # --------------------------------------------------------

    X_train_aligned = align_polarity(X_train)
    X_cal_aligned = align_polarity(X_cal)
    X_eval_aligned = align_polarity(X_eval)

    print("\nInput dtype:")
    print("Baseline:", X_train.dtype)
    print("Aligned:", X_train_aligned.dtype)

    # --------------------------------------------------------
    # Run experiments
    # --------------------------------------------------------

    results = []

    results.append(
        run_experiment(
            name="Baseline",
            X_train=X_train,
            X_cal=X_cal,
            y_cal=y_cal,
            X_eval=X_eval,
            y_eval=y_eval,
            eval_symbols=eval_symbols
        )
    )

    results.append(
        run_experiment(
            name="Polarity Aligned",
            X_train=X_train_aligned,
            X_cal=X_cal_aligned,
            y_cal=y_cal,
            X_eval=X_eval_aligned,
            y_eval=y_eval,
            eval_symbols=eval_symbols
        )
    )

    # --------------------------------------------------------
    # Comparison
    # --------------------------------------------------------

    df = pd.DataFrame(results)

    df.to_csv(
        RESULT_DIR / "comparison.csv",
        index=False
    )

    metrics = [
        "precision",
        "recall",
        "f1",
        "ap",
        "roc_auc"
    ]

    plot_df = (
        df.set_index("method")[metrics].T
    )

    ax = plot_df.plot.bar(
        figsize=(10, 5),
        rot=0
    )

    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")

    ax.set_title(
        "LSTM Autoencoder - Polarity Comparison"
    )

    ax.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        RESULT_DIR / "comparison.png",
        dpi=200
    )

    plt.close()

    # --------------------------------------------------------
    # Save configuration
    # --------------------------------------------------------

    config = {
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "hidden_size": HIDDEN_SIZE,
        "latent_size": LATENT_SIZE,
        "learning_rate": LR,
        "max_epochs": EPOCHS,
        "min_epochs": MIN_EPOCHS,
        "patience": PATIENCE,
        "gradient_clip": GRAD_CLIP,
        "calibration_groups": sorted(CAL_GROUPS),
        "evaluation_groups": sorted(EVAL_GROUPS),
        "test_data_used": False,
        "experiment_type": "exploratory"
    }

    with open(
        RESULT_DIR / "config.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            config,
            f,
            indent=4
        )

    print("\n========== FINAL COMPARISON ==========")

    print(
        df.to_string(index=False)
    )

    print(
        "\nResults saved:",
        RESULT_DIR.resolve()
    )


if __name__ == "__main__":
    main()
