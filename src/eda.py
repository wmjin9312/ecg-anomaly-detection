
from pathlib import Path
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
import wfdb

DATA_PATH = "data/raw/mitdb/100"
SAVE_DIR = Path("results/figures")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# 1. ECG 신호와 어노테이션 로딩
record = wfdb.rdrecord(DATA_PATH)
annotation = wfdb.rdann(DATA_PATH, "atr")

fs = record.fs
signal = record.p_signal[:, 0]  # MLII 채널

# 2. 박동 유형별 개수 확인
counts = Counter(annotation.symbol)

print("\n=== Annotation Distribution ===")
for symbol, count in counts.most_common():
    print(f"{symbol}: {count}")

# 3. 10초 ECG 신호 추출
start_sec = 0
end_sec = 10

start = int(start_sec * fs)
end = int(end_sec * fs)

time = np.arange(start, end) / fs
ecg = signal[start:end]

# 4. 표시할 어노테이션 위치
mask = (
    (annotation.sample >= start)
    & (annotation.sample < end)
)

beat_positions = annotation.sample[mask]
beat_symbols = np.array(annotation.symbol)[mask]

# 5. ECG 파형 시각화
plt.figure(figsize=(15, 5))
plt.plot(time, ecg, linewidth=1, color="black", label="ECG")

for pos, symbol in zip(beat_positions, beat_symbols):
    if symbol not in {"N", "A", "V", "L", "R"}:
        continue

    x = pos / fs
    y = signal[pos]

    color = "red" if symbol != "N" else "green"

    plt.scatter(x, y, color=color, s=35, zorder=3)
    plt.text(x, y + 0.12, symbol, ha="center", color=color)

plt.title("MIT-BIH Record 100 - ECG Signal (10 seconds)")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude (mV)")
plt.grid(alpha=0.3)
plt.tight_layout()

save_path = SAVE_DIR / "ecg_signal_10sec.png"
plt.savefig(save_path, dpi=200)
plt.show()

print(f"\nSaved: {save_path}")
