# Individual Project Proposal: Real-Time Edge Acoustic Anomaly Detection

**Student Name:Tadiwanashe Munemo

**Student ID:70712 

## 1. Problem Statement
Industrial rotating machinery (pumps, HVAC compressors, and conveyor motors) exhibits subtle acoustic shifts prior to mechanical failure. Continuous cloud streaming of raw audio is bandwidth-prohibitive and introduces network latency. This project proposes a lightweight, real-time acoustic anomaly detection model deployed directly on edge hardware to classify normal vs. anomalous machine operating states from short audio windows.

## 2. Dataset & Preprocessing Pipeline
* **Dataset:** **MIMII Dataset** (Malfunctioning Industrial Machine Investigation and Inspection) — specifically the industrial fan and pump subsets (16 kHz mono WAV recordings at 0 dB and 6 dB SNR).
* **Split Strategy:** Stratified 80% / 10% / 10% (Train / Validation / Test) across distinct machine IDs to prevent acoustic background leakage.

## 3. Success Metrics
* **Primary Classification Metric:** **Macro F1-Score $\ge 0.92$** and **ROC-AUC $\ge 0.96$** on unseen anomalous test samples (critical due to natural class imbalance between normal operation and fault events).
* **Operational Reliability:** False Positive Rate (FPR) $< 3\%$ at $95\%$ True Positive Rate (TPR) to avoid nuisance maintenance alerts.

## 4. Deployment Constraints Compliance
* **Inference Latency:** Using depthwise-separable 2D convolutions (MobileNet-style inverted residual blocks) keeps compute under **12 MFLOPs** per spectrogram frame, achieving **~6–11 ms** single-thread CPU inference (well below the 50 ms budget).
* **Model Footprint:** Uncompressed FP32 baseline will target ~320k parameters.

## 5. Planned Optimization Strategy
1. **Structured Channel Pruning:** Prune redundant filters in intermediate depthwise-separable layers using L1-norm channel importance scoring, followed by 5 epochs of fine-tuning to recover accuracy.
2. **Post-Training Static Quantization:** Convert weights and activations from FP32 to INT8 via PyTorch FX Graph Mode Quantization, reducing model size by ~4x and accelerating CPU matrix multiplications by 2–3x.
3. **Export & Runtime Validation:** Export the quantized graph to TorchScript / ONNX Runtime and benchmark latency and peak RSS memory over 1,000 consecutive inference iterations.
