
import wfdb

path = "data/raw/mitdb/100"

record = wfdb.rdrecord(path)
annotation = wfdb.rdann(path, "atr")

print("샘플링 주파수:", record.fs)
print("신호 형태:", record.p_signal.shape)
print("ECG 채널:", record.sig_name)
print("어노테이션 개수:", len(annotation.sample))
print("처음 20개 심박 유형:", annotation.symbol[:20])
