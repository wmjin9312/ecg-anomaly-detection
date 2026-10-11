# ECG Anomaly Detection: Isolation Forest vs. LSTM Autoencoder

**MIT-BIH Arrhythmia Database 기반 심박 단위 이상 탐지 및 파형 극성 정렬 실험**

Python · PyTorch · Scikit-learn · WFDB · ECG Time Series · Anomaly Detection

> **Project summary**  
> 정상 계열 심박만 학습한 Isolation Forest와 LSTM Autoencoder의 이상 탐지 성능을 비교하고, 오류가 집중되는 기록과 박동 유형을 분석했습니다. 특히 좌각차단형(L) 박동의 기록 간 파형 차이를 관찰한 뒤, QRS 극성 정렬 전처리가 탐지 성능에 미치는 영향을 탐색적으로 평가했습니다.
>
> **범위 안내:** 연구·학습 목적의 오프라인 실험입니다. 의료 진단이나 실시간 환자 모니터링용으로 검증된 시스템이 아닙니다.

## Visual Highlights

실제 실험에서 생성한 그래프입니다. 아래 이미지를 클릭하면 원본 크기로 볼 수 있습니다.

| ECG signal & annotations | L-beat waveform distribution shift |
|:---:|:---:|
| [![ECG signal with annotations](results/figures/ecg_signal_10sec.png)](results/figures/ecg_signal_10sec.png) | [![Mean L-beat waveforms, Train vs Test](results/l_beat_analysis/mean_l_waveforms.png)](results/l_beat_analysis/mean_l_waveforms.png) |
| **Record 100:** 실제 ECG와 박동 주석 | **Records 109/111 vs 207:** L 박동 평균 파형 차이 |

| Anomaly detection by beat type | Polarity alignment comparison |
|:---:|:---:|
| [![Anomaly recall by AAMI beat type](results/error_analysis/beat_type_recall.png)](results/error_analysis/beat_type_recall.png) | [![LSTM-AE polarity alignment metrics](results/lstm_polarity_experiment/comparison.png)](results/lstm_polarity_experiment/comparison.png) |
| **S/V/F:** 이상 박동 유형별 탐지율 | **LSTM-AE:** 극성 정렬 전후 Evaluation 성능 |

> 이 시각화는 **초기 Test 분석**과 **Validation 기반 탐색적 개선 실험**을 함께 보여주는 안내판입니다. 두 평가 결과는 동일한 데이터셋의 성능 숫자가 아니므로 아래 표를 구분해 해석해 주세요.

---

## 1. Project Overview

### 문제 정의

ECG는 환자별 파형 특성과 시간적 구조가 다른 시계열 신호입니다. 실제 이상 박동의 라벨이 부족한 상황을 가정하여, **정상 계열 박동만으로 모델을 학습한 뒤** 새로운 심박에 이상 점수를 부여하는 방식을 실험했습니다.

핵심 질문은 다음과 같습니다.

1. 정상 박동만 학습하는 **Isolation Forest**와 **LSTM Autoencoder**는 새로운 기록의 이상 박동을 얼마나 구분하는가?
2. 오탐(False Positive)과 미탐(False Negative)은 특정 기록이나 심박 유형에 집중되는가?
3. ECG 파형의 **극성 정렬(Polarity Alignment)**이 모델의 이상 탐지 성능에 영향을 주는가?

### 실험 흐름

```text
MIT-BIH ECG records (48)
        |
        v
WFDB 로딩 / annotation filtering
        |
        v
심박 중심 234-sample 추출 + beat-wise Z-score
        |
        v
기록 그룹 기준 Train / Validation / Test 분할
        |
        +-----------------------------+
        |                             |
        v                             v
Isolation Forest              LSTM Autoencoder
        |                             |
        +--------------+--------------+
                       v
             이상 점수 및 임계값
                       v
           Precision / Recall / F1
                AP / ROC-AUC
                       v
         기록별 · 박동 유형별 오류 분석
                       v
        Polarity Alignment 탐색 실험
```

## 2. Dataset & Preprocessing

**Dataset:** [MIT-BIH Arrhythmia Database, PhysioNet (v1.0.0)](https://physionet.org/content/mitdb/1.0.0/)

- 총 **48개 ECG 기록**, 기록별 **650,000 시점**, **2채널**, **360 Hz**
- 전문가의 박동 위치 및 유형 어노테이션(`.atr`) 사용
- 이번 구현에서는 각 기록의 **첫 번째 신호 채널**을 사용했습니다. 기록별 첫 번째 채널의 유도 종류가 동일하지 않을 수 있습니다.
- 각 박동 주석 위치를 기준으로 **앞 90개 + 뒤 144개 샘플**, 총 **234개 시점**을 추출했습니다.
- 추출한 각 박동은 개별 평균과 표준편차를 이용해 Z-score 정규화했습니다.

### 라벨 정의

AAMI 기반의 박동 그룹 구성을 참고해 아래와 같이 **이진 이상 탐지 과제**를 정의했습니다.

| 그룹 | 어노테이션 심볼 | 이 실험의 라벨 |
|---|---|---|
| N 계열 | `N`, `L`, `R`, `e`, `j` | 0 (정상 그룹) |
| S 계열 | `A`, `a`, `J`, `S` | 1 (이상 그룹) |
| V 계열 | `V`, `E` | 1 (이상 그룹) |
| F 계열 | `F` | 1 (이상 그룹) |
| 제외 | `/`, `f`, `Q` 등 기타/비박동 주석 | 평가 제외 |

**주의:** 여기서 '정상 그룹'은 **이 실험에서 정의한 분류상의 N 계열**을 뜻합니다. `L`(좌각차단형), `R`(우각차단형)이 의학적으로 정상 동성 박동과 같다는 뜻이 아닙니다. 또한 제외된 유형은 모델의 탐지 범위에 포함되지 않습니다.

### 데이터 분할 및 누수 방지

기록 단위 그룹으로 데이터를 분할했고, 알려진 동일 대상자의 기록 `201`과 `202`는 하나의 그룹(`201_202`)으로 묶었습니다. 분할 사이에 그룹 중복이 없는지 검사했습니다.

| Split | 샘플 수 | 정상 | 이상 | 용도 |
|---|---:|---:|---:|---|
| Train | 61,154 | 61,154 | 0 | 정상 데이터만 학습 |
| Validation | 16,939 | 15,591 | 1,348 | 체크포인트 및 임계값 선정 |
| Test | 15,410 | 13,850 | 1,560 | 최초 모델의 별도 평가 |

기록별 차이로 인해 환자 그룹 간 일반화 성능에 편차가 발생할 수 있으며, 이번 분할은 표준 AAMI DS1/DS2 평가와 동일하지 않습니다.

### ECG 시각화

![Record 100 ECG signal](results/figures/ecg_signal_10sec.png)

## 3. Models & Evaluation

### Isolation Forest

- 정상 박동의 234개 값을 입력 벡터로 사용
- `n_estimators=200`, `max_samples=256`, `random_state=42`
- `-score_samples()`를 사용해 값이 클수록 이상도가 높도록 정의

### LSTM Autoencoder

- 입력: `(batch, 234, 1)`
- LSTM Encoder → 32차원 Latent Representation → LSTM Decoder
- Hidden Size 64, Reconstruction MSE, Adam Optimizer
- 정상 박동만으로 학습하고 입력-복원 신호 간 MSE를 이상 점수로 사용

두 모델 모두 **Validation 라벨을 사용해 F1 최대 임계값을 선택**했습니다. 즉, 모델 학습은 정상 중심이지만 임계값 선정은 라벨이 포함된 검증 데이터에 의존하는 방식입니다.

평가 지표: **Precision, Recall, F1, Average Precision (AP), ROC-AUC**. 본 프로젝트의 `test_pr_auc` 필드는 `average_precision_score` 계산값이므로 보고서에서는 **AP**라고 표기합니다.

## 4. Initial Held-out Test Results

다음은 **최초 모델 구현과 전처리 설정에 대한 Test 15,410개** 결과입니다. 이후 개선 실험의 평가 결과와 혼합하지 않았습니다.

| Model | Precision | Recall | F1 | AP | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| **Isolation Forest** | **0.3510** | **0.5519** | **0.4291** | **0.4657** | **0.8813** |
| LSTM Autoencoder | 0.3026 | 0.3891 | 0.3404 | 0.4244 | 0.7952 |

Isolation Forest가 이번 Test에서 모든 주요 지표에서 앞섰습니다. 다만 이는 **해당 분할과 실험 조건에서의 관찰 결과**이며, 일반적으로 단순한 모델이 우월하다는 의미는 아닙니다.

### Confusion Matrices — Initial Test

| Isolation Forest | LSTM Autoencoder |
|:---:|:---:|
| ![Isolation Forest confusion matrix](results/isolation_forest/confusion_matrix.png) | ![LSTM Autoencoder confusion matrix](results/lstm_autoencoder/confusion_matrix.png) |

초기 Test에서 Isolation Forest는 FP 1,592개, FN 699개를 기록했고, LSTM-AE는 FP 1,399개, FN 953개를 기록했습니다.

### Isolation Forest — ROC / PR curves

| ROC curve | Precision–Recall curve |
|:---:|:---:|
| ![Isolation Forest ROC curve](results/isolation_forest/roc_curve.png) | ![Isolation Forest PR curve](results/isolation_forest/pr_curve.png) |

## 5. Error Analysis

### 기록별 오류 집중: Record 207

오탐 분석에서 Record `207`이 핵심 사례로 나타났습니다.

| Model | 전체 Test FP | Record 207 FP | Record 207에서 L 박동 FP |
|---|---:|---:|---:|
| Isolation Forest | 1,592 | 1,434 | **1,432 / 1,457** |
| LSTM Autoencoder | 1,399 | 1,303 | **1,303 / 1,457** |

Record 207에서 N 계열로 분류된 좌각차단형(`L`) 박동에 오탐이 집중됐습니다. 훈련 데이터의 L 박동 4,614개는 Record `109`와 `111`에서만 유래했으며, Test Record `207`의 평균 L 파형은 주요 QRS 편향의 **극성 방향이 다르게 관찰**됐습니다.

![Mean L-beat waveforms across records](results/l_beat_analysis/mean_l_waveforms.png)

이는 **기록/환자 간 파형 분포 차이**가 탐지 오류와 관련될 수 있다는 가설을 뒷받침하지만, **극성 차이가 오류의 유일하거나 직접적인 원인임을 입증하지는 않습니다.**

### 이상 박동 유형별 Recall (Initial Test)

| AAMI 이상 그룹 | Test 샘플 수 | Isolation Forest | LSTM-AE |
|---|---:|---:|---:|
| S | 193 | 0.5700 | 0.5648 |
| V | 1,341 | 0.5585 | 0.3706 |
| F | 26 | 0.0769 | 0.0385 |

특히 F 그룹은 표본이 26개뿐이므로 일반적인 성능을 단정하기 어렵습니다.

![Anomaly recall by beat group](results/error_analysis/beat_type_recall.png)

![Patient group F1](results/error_analysis/patient_f1.png)

### Misclassification Examples — Initial Test

아래는 최초 Test에서 Isolation Forest 기준으로 추출한 대표 **오탐(FP)**, **미탐(FN)**, **두 모델이 모두 틀린 심박**입니다. 각 그림은 정규화한 파형의 처음 5개 사례를 보여주며, 전체 오류를 대표하는 통계적 표본은 아닙니다.

| False Positives | False Negatives |
|:---:|:---:|
| ![False-positive ECG beats](results/error_analysis/false_positive.png) | ![False-negative ECG beats](results/error_analysis/false_negative.png) |

**Both models wrong**

![ECG beats misclassified by both models](results/error_analysis/both_wrong.png)


## 6. Exploratory Improvement: Polarity Alignment

### 가설 및 방법

**가설:** 박동별 QRS 주요 편향의 극성을 일관된 방향으로 정렬하면, 환자 간 신호 표현 차이 일부를 줄여 탐지 성능이 달라질 수 있다.

- 기존 234-sample 표준화 파형의 R-peak 중심 부근(인덱스 90 ± 35)에서 **절댓값이 가장 큰 샘플**을 찾습니다.
- 해당 샘플의 값이 음수이면 **박동 전체에 −1을 곱합니다**.
- 두 모델에 동일하게 적용하고, Baseline과 극성 정렬 조건을 각각 비교했습니다.

이 변환은 유용한 극성 정보를 제거할 수도 있습니다. 따라서 모든 의료 시계열에 일반적으로 적용할 수 있는 방법으로 주장하지 않습니다.

### 평가 프로토콜

최초 Test를 사후 분석한 뒤 개선 가설을 세웠으므로, 기존 Test를 성능 개선 선택에 다시 사용하지 않았습니다. 대신 **기존 Validation의 기록 그룹을 Calibration과 Evaluation으로 나누어** 탐색적 실험을 수행했습니다.

- **Calibration:** `203`, `214`, `231` — 체크포인트 및 임계값 선정
- **Evaluation:** `108`, `113`, `201_202`, `222` — 결과 비교
- Evaluation은 **10,135개 심박**, 이 중 이상 641개
- 두 모델 모두 같은 Calibration/Evaluation 그룹을 사용했습니다.
- 이 Evaluation에는 `L` 박동이 **0개**입니다. 따라서 이 실험이 Record 207의 L 오탐을 해결했는지는 평가하지 못했습니다.

### 동일 Evaluation 그룹의 4가지 조건 비교

| Model | Preprocessing | Precision | Recall | F1 | AP | ROC-AUC |
|---|---|---:|---:|---:|---:|---:|
| Isolation Forest | Baseline | 0.0919 | 0.2543 | 0.1350 | 0.1066 | 0.7221 |
| Isolation Forest | **Polarity Aligned** | **0.1750** | **0.6178** | **0.2727** | **0.2629** | **0.7901** |
| LSTM Autoencoder | Baseline | 0.1013 | 0.6365 | 0.1748 | 0.0799 | 0.6394 |
| LSTM Autoencoder | **Polarity Aligned** | **0.1629** | **0.6708** | **0.2622** | **0.1503** | **0.7786** |

두 모델 모두 이 Evaluation 그룹에서 극성 정렬 적용 후 F1과 AP가 높아졌습니다. LSTM-AE의 FP는 **3,619 → 2,209**로 줄었고 TP는 **408 → 430**으로 늘었습니다.

![LSTM polarity experiment comparison](results/lstm_polarity_experiment/comparison.png)

### LSTM Autoencoder Training Curves

| Baseline | Polarity Aligned |
|:---:|:---:|
| ![LSTM-AE baseline training curve](results/lstm_polarity_experiment/baseline_training.png) | ![LSTM-AE polarity-aligned training curve](results/lstm_polarity_experiment/polarity_aligned_training.png) |

### LSTM Autoencoder Confusion Matrices — Exploratory Evaluation

| Baseline | Polarity Aligned |
|:---:|:---:|
| ![Baseline evaluation confusion matrix](results/lstm_polarity_experiment/baseline_confusion_matrix.png) | ![Aligned evaluation confusion matrix](results/lstm_polarity_experiment/polarity_aligned_confusion_matrix.png) |


### 반복 분할 민감도 분석: Isolation Forest

Validation 그룹에서 10회의 GroupShuffleSplit을 수행한 탐색적 비교에서는 평균 F1이 **0.1994 → 0.2996**, 평균 AP가 **0.1861 → 0.2939**으로 변화했습니다.

**해석 주의:** 동일한 제한된 Validation 그룹을 반복 재사용한 결과이며, 독립된 10개 외부 데이터셋에서 검증한 것이 아닙니다. 따라서 이 수치로 통계적 유의성이나 일반화된 성능 향상을 주장하지 않습니다. LSTM-AE 극성 정렬 비교도 단일 고정 시드에서 수행했습니다.

## 7. Key Findings & Limitations

**관찰된 결과**

- 최초 별도 Test에서는 Isolation Forest가 LSTM-AE보다 높은 F1, AP, ROC-AUC를 기록했습니다.
- 오탐은 Record 207의 `L` 박동에 집중됐습니다.
- 훈련과 해당 테스트 기록의 평균 L 파형 사이에서 극성 차이가 관찰됐습니다.
- 탐색적 Calibration/Evaluation 비교에서는 극성 정렬 후 두 모델의 F1과 AP가 개선됐습니다.

**한계**

- 개선 방법은 초기 Test 오류 분석에서 도출했으므로 **완전히 독립적인 최종 검증이 아닙니다**.
- 탐색적 Evaluation에는 `L` 박동이 없어 원래의 L 박동 오탐 가설을 직접 검증하지 못했습니다.
- 환자/기록 수가 제한적이고, 환자 간 파형·유도 차이에 민감할 수 있습니다.
- 첫 번째 채널의 ECG 유도 종류가 기록마다 동일하지 않습니다.
- `/`, `f`, Q 계열 등은 평가에서 제외했습니다.
- LSTM-AE 비교는 단일 시드이며, 학습 결과에 초기화·최적화 변동성이 존재할 수 있습니다.
- 실제 의료 환경에서의 경고 비용, 실시간 성능 및 임상적 유효성을 검증하지 않았습니다.

**후속 과제:** 사전 정의된 새로운 환자 그룹 기반 외부 평가, 여러 초기화 시드 반복, 유도별 전처리 비교, 기록별 이상 점수 분석, 알람 수·오탐 비용 기반의 운영 평가.

## 8. Repository Structure

```text
ecg-anomaly-detection/
├── README.md
├── requirements.txt
├── src/
│   ├── download.py
│   ├── inspect_data.py
│   ├── verify_dataset.py
│   ├── eda.py
│   ├── preprocess.py
│   ├── isolation_forest.py
│   ├── lstm_autoencoder.py
│   ├── error_analysis.py
│   ├── check_annotations.py
│   ├── analyze_l_beats.py
│   ├── polarity_experiment.py
│   ├── polarity_robustness.py
│   └── lstm_polarity_experiment.py
├── data/                      # Git 관리 제외
│   ├── raw/mitdb/
│   └── processed/
└── results/
    ├── figures/
    ├── isolation_forest/
    ├── lstm_autoencoder/
    ├── error_analysis/
    ├── l_beat_analysis/
    ├── polarity_experiment/
    ├── polarity_robustness/
    └── lstm_polarity_experiment/
```

학습된 `.pt` 가중치와 원본/가공 데이터는 기본적으로 저장소에 포함하지 않습니다. README의 그래프와 CSV/JSON 결과는 예시가 아니라 **실제 수행한 실험 결과**입니다.

## 9. How to Reproduce

**환경:** Windows · Python 3.14.3 · PyTorch 2.14.1+cu130 · NVIDIA RTX 3070 (실험에 사용한 환경)

### ① Repository 및 Python 환경

```powershell
git clone https://github.com/wmjin9312/ecg-anomaly-detection.git
cd ecg-anomaly-detection
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

`requirements.txt`는 개발 환경에서 `pip freeze`로 생성한 파일입니다. CUDA 지원 PyTorch 설치는 OS·GPU·Python 버전에 따라 달라질 수 있으므로 [PyTorch 공식 설치 안내](https://pytorch.org/get-started/locally/)를 확인하세요. `torch.cuda.is_available()`로 GPU 사용 가능 여부를 확인할 수 있습니다.

### ② 데이터 다운로드

PhysioNet의 [MIT-BIH 데이터셋](https://physionet.org/content/mitdb/1.0.0/)을 내려받아 아래 위치에 `.dat`, `.hea`, `.atr` 파일을 배치합니다.

```text
data/raw/mitdb/100.dat
data/raw/mitdb/100.hea
data/raw/mitdb/100.atr
...
```

공개 S3 버킷 접근이 가능한 경우 AWS CLI를 이용할 수도 있습니다.

```powershell
aws s3 sync s3://physionet-open/mitdb/1.0.0/ .\data\raw\mitdb\ --no-sign-request
```

### ③ 데이터 검증·전처리 및 최초 모델 평가

```powershell
python src/verify_dataset.py
python src/eda.py
python src/preprocess.py
python src/isolation_forest.py
python src/lstm_autoencoder.py
```

### ④ 사후 오류 분석 및 개선 실험

```powershell
python src/error_analysis.py
python src/check_annotations.py
python src/analyze_l_beats.py
python src/polarity_experiment.py
python src/polarity_robustness.py
python src/lstm_polarity_experiment.py
```

일부 스크립트는 앞 단계에서 생성한 `data/processed/*.npz`와 로컬 예측 파일에 의존합니다. CUDA 지원 모델의 재학습은 환경과 초기화에 따라 결과가 조금씩 달라질 수 있습니다.

## 10. Reference & Acknowledgment

- [MIT-BIH Arrhythmia Database — PhysioNet](https://physionet.org/content/mitdb/1.0.0/)
- [WFDB Python](https://wfdb.readthedocs.io/)
- [Scikit-learn Isolation Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)
- [PyTorch LSTM](https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html)

본 프로젝트는 공개 ECG 데이터에 대한 개인 연구·포트폴리오 프로젝트이며, 의료기기 또는 임상적 의사결정 시스템이 아닙니다.
