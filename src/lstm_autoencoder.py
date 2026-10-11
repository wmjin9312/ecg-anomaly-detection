
from pathlib import Path
import json
import random
import copy

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

SEED = 42
BATCH_SIZE = 128
HIDDEN_SIZE = 64
LATENT_SIZE = 32
EPOCHS = 30
PATIENCE = 7
LR = 1e-3

DATA_DIR = Path("data/processed")
RESULT_DIR = Path("results/lstm_autoencoder")
RESULT_DIR.mkdir(parents=True, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", DEVICE)

if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


class LSTMAutoencoder(nn.Module):
    def __init__(self, hidden_size=64, latent_size=32):
        super().__init__()

        self.encoder = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            batch_first=True,
        )

        self.to_latent = nn.Linear(
            hidden_size, latent_size
        )

        self.decoder = nn.LSTM(
            input_size=latent_size,
            hidden_size=hidden_size,
            batch_first=True,
        )

        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, (hidden, _) = self.encoder(x)

        z = self.to_latent(hidden[-1])

        decoder_input = z.unsqueeze(1).expand(
            -1, x.size(1), -1
        )

        decoded, _ = self.decoder(decoder_input)

        return self.output_layer(decoded)


def load_data(name):
    with np.load(DATA_DIR / f"{name}.npz") as data:
        X = data["X"].astype(np.float32)
        y = data["y"].astype(np.int64)

    return X[:, :, None], y


def make_loader(X, batch_size, shuffle=False):
    dataset = TensorDataset(torch.from_numpy(X))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
    )


def epoch_loss(model, loader, optimizer=None):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    count = 0
    criterion = nn.MSELoss()

    context = torch.enable_grad() if training else torch.no_grad()

    with context:
        for (batch,) in loader:
            batch = batch.to(DEVICE, non_blocking=True)

            if training:
                optimizer.zero_grad(set_to_none=True)

            reconstruction = model(batch)
            loss = criterion(reconstruction, batch)

            if training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(batch)
            count += len(batch)

    return total_loss / count


@torch.no_grad()
def anomaly_scores(model, X):
    model.eval()

    loader = make_loader(X, BATCH_SIZE)
    scores = []

    for (batch,) in loader:
        batch = batch.to(DEVICE, non_blocking=True)
        reconstructed = model(batch)

        errors = ((batch - reconstructed) ** 2).mean(
            dim=(1, 2)
        )

        scores.append(errors.cpu().numpy())

    return np.concatenate(scores)


def select_threshold(y_true, scores):
    precision, recall, thresholds = precision_recall_curve(
        y_true, scores
    )

    f1 = (
        2 * precision[:-1] * recall[:-1]
        / (precision[:-1] + recall[:-1] + 1e-12)
    )

    idx = int(np.argmax(f1))
    return float(thresholds[idx]), float(f1[idx])


def main():
    X_train, y_train = load_data("train_normal")
    X_val, y_val = load_data("validation")
    X_test, y_test = load_data("test")

    assert np.all(y_train == 0)
    assert set(np.unique(y_val)) == {0, 1}
    assert set(np.unique(y_test)) == {0, 1}

    print("Train:", X_train.shape)
    print("Validation:", X_val.shape)
    print("Test:", X_test.shape)

    train_loader = make_loader(
        X_train, BATCH_SIZE, shuffle=True
    )

    # 정상 Validation 샘플로 재구성 성능 확인
    val_normal_loader = make_loader(
        X_val[y_val == 0], BATCH_SIZE
    )

    model = LSTMAutoencoder(
        hidden_size=HIDDEN_SIZE,
        latent_size=LATENT_SIZE,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR
    )

    best_loss = float("inf")
    best_state = None
    patience_count = 0
    history = {"train": [], "val": []}

    print("\nTraining LSTM Autoencoder...")

    for epoch in range(1, EPOCHS + 1):
        train_loss = epoch_loss(
            model, train_loader, optimizer
        )

        val_loss = epoch_loss(
            model, val_normal_loader
        )

        history["train"].append(train_loss)
        history["val"].append(val_loss)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f}"
        )

        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_count = 0
        else:
            patience_count += 1

        if patience_count >= PATIENCE:
            print("Early stopping triggered")
            break

    model.load_state_dict(best_state)

    torch.save(
        {
            "model_state_dict": best_state,
            "hidden_size": HIDDEN_SIZE,
            "latent_size": LATENT_SIZE,
        },
        RESULT_DIR / "best_model.pt"
    )

    # Validation에서 이상 탐지 임계값 결정
    val_scores = anomaly_scores(model, X_val)
    threshold, val_f1 = select_threshold(
        y_val, val_scores
    )

    # 고정된 모델과 임계값으로 Test 평가
    test_scores = anomaly_scores(model, X_test)
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
        "test_average_precision": float(
            average_precision_score(y_test, test_scores)
        ),
        "test_roc_auc": float(
            roc_auc_score(y_test, test_scores)
        ),
    }

    print("\n========== TEST RESULTS ==========")

    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    cm = confusion_matrix(
        y_test, y_pred, labels=[0, 1]
    )

    print("\nConfusion Matrix:")
    print(cm)

    with open(
        RESULT_DIR / "metrics.json",
        "w", encoding="utf-8"
    ) as f:
        json.dump(metrics, f, indent=4)

    np.savez_compressed(
        RESULT_DIR / "test_predictions.npz",
        y_true=y_test,
        y_pred=y_pred,
        scores=test_scores,
    )

    # 학습 곡선
    plt.figure(figsize=(8, 5))
    plt.plot(history["train"], label="Train Loss")
    plt.plot(history["val"], label="Val Normal Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("LSTM Autoencoder Training")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "training_curve.png", dpi=200)
    plt.close()

    # Confusion Matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Normal", "Abnormal"],
    ).plot(ax=ax, cmap="Blues", colorbar=False)
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "confusion_matrix.png", dpi=200)
    plt.close()

    # 이상 점수 분포
    plt.figure(figsize=(8, 5))
    plt.hist(
        test_scores[y_test == 0],
        bins=80,
        alpha=0.6,
        density=True,
        label="Normal",
    )
    plt.hist(
        test_scores[y_test == 1],
        bins=80,
        alpha=0.6,
        density=True,
        label="Abnormal",
    )
    plt.axvline(
        threshold, linestyle="--",
        color="red", label="Threshold"
    )
    plt.xlabel("Reconstruction Error")
    plt.ylabel("Density")
    plt.title("Anomaly Score Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        RESULT_DIR / "anomaly_distribution.png",
        dpi=200
    )
    plt.close()

    print("\nResults saved:", RESULT_DIR.resolve())


if __name__ == "__main__":
    main()
