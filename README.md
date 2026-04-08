# 🚀 Mobile-Efficient Image Classification: NAS + Structured Pruning on CIFAR-10

A comprehensive pipeline for building edge-deployable image classifiers by combining **MobileNetV2**, **MobileNetV3**, **DARTS-based Neural Architecture Search**, and **Structured Pruning** — targeting efficient inference on resource-constrained devices like the Raspberry Pi.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture Pipeline](#architecture-pipeline)
- [Models](#models)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Evaluation Metrics](#evaluation-metrics)
- [ONNX Export & Edge Deployment](#onnx-export--edge-deployment)
- [Configuration](#configuration)
- [Results](#results)
- [Tech Stack](#tech-stack)

---

## Overview

This project investigates the synergy between **pre-trained efficient backbones** and **automated compression techniques** for CIFAR-10 classification. It benchmarks four distinct model strategies:

1. **Baseline MobileNetV2** — ImageNet-pretrained, fine-tuned on CIFAR-10
2. **Pruned MobileNetV2** — Structured channel pruning + fine-tuning
3. **DARTS Final** — Custom micro-architecture discovered via differentiable NAS
4. **DARTS + Pruned** — NAS-derived architecture further compressed with pruning

All four variants are exported to **12 ONNX model files** (FP32 / INT8 / Optimised) and benchmarked end-to-end for latency, throughput, and accuracy — making them ready for real-world edge deployment.

---

## Key Features

- **Multi-backbone support** — MobileNetV2 and MobileNetV3 baselines with CIFAR-10 head adaptation
- **Structured channel pruning** — L1-norm-based filter removal with configurable sparsity; depthwise and first/last layers optionally skipped for stability
- **DARTS (Differentiable Architecture Search)** — Bi-level optimisation with learnable architecture parameters (alphas) over a mixed-operation space
- **12-variant ONNX export** — FP32, INT8 dynamic quantisation, and graph-optimised "Prep" variants for every model
- **Edge benchmarking suite** — Measures avg/min/max/P95 latency, FPS, speedup, and parameter count on both Colab and Raspberry Pi
- **Comprehensive visualisation** — Side-by-side accuracy, parameter count, inference time, and FLOPs plots across all variants

---

## Architecture Pipeline

```
CIFAR-10 Dataset
      │
      ├─── MobileNetV2 / MobileNetV3 (Pretrained)
      │           │
      │     Fine-tune on CIFAR-10
      │           │
      │     Structured Pruning (30% filters)
      │           │
      │     Fine-tune Pruned Model
      │
      └─── DARTS Search Network
                  │
            Architecture Search
            (Bi-level Optimisation)
                  │
            Discrete Genotype
                  │
            Train Final Network
                  │
            Structured Pruning
                  │
            Fine-tune Pruned DARTS

All 4 variants ──► ONNX Export (FP32 / INT8 / Optimised)
                        │
                 Edge Benchmark (RPi / Colab)
```

---

## Models

### MobileNetV2 (Baseline)
- ImageNet pre-trained backbone from `torchvision`
- Final classifier replaced with `Linear(1280 → 10)`
- Trained for 50 epochs with SGD + cosine-style MultiStepLR

### MobileNetV3
- Lightweight successor with hard-swish activations and SE blocks
- Same CIFAR-10 head adaptation strategy as V2
- Used as an additional efficient baseline for comparison

### Pruned MobileNetV2
- L1-norm structured pruning at 30% filter sparsity
- Depthwise convolutions and stem/head layers optionally preserved
- 30-epoch fine-tuning after pruning to recover accuracy

### DARTS Final Network
- Search space: SepConv 3×3, SepConv 5×5, DilConv 3×3, DilConv 5×5, MaxPool, AvgPool, Skip, Zero
- 6-layer search network with 16 initial channels
- Bi-level optimisation: network weights (SGD) + architecture alphas (Adam)
- Discrete genotype derived via argmax of learned alphas

### DARTS + Pruned
- Pruning applied on top of the DARTS-derived final network
- Combined NAS + compression for maximum efficiency

---

## Dataset

**CIFAR-10** — 60,000 32×32 colour images, 10 classes

| Split      | Size   |
|------------|--------|
| Train      | 40,000 |
| Validation | 10,000 |
| Test       | 10,000 |

**Augmentation (Training):**
- RandomCrop (32, padding=4)
- RandomHorizontalFlip
- Normalisation: mean `(0.4914, 0.4822, 0.4465)`, std `(0.2023, 0.1994, 0.2010)`

---

## Project Structure

```
.
├── MobilneNETV2FInal.ipynb     # Main notebook — all cells in sequence
├── models/                     # Saved PyTorch .pth checkpoints
│   ├── baseline_mobilenetv2.pth
│   ├── pruned_mobilenetv2_finetuned.pth
│   ├── darts_search.pth
│   ├── darts_final.pth
│   └── darts_pruned.pth
├── onnx_edge_models/           # Exported ONNX files (12 variants)
│   ├── baseline_mobilenet_fp32.onnx
│   ├── baseline_mobilenet_int8.onnx
│   ├── baseline_mobilenet_prep.onnx
│   ├── pruned_mobilenet_fp32.onnx
│   ├── pruned_mobilenet_int8.onnx
│   ├── pruned_mobilenet_prep.onnx
│   ├── darts_final_fp32.onnx
│   ├── darts_final_int8.onnx
│   ├── darts_final_prep.onnx
│   ├── darts_pruned_fp32.onnx
│   ├── darts_pruned_int8.onnx
│   ├── darts_pruned_prep.onnx
│   └── manifest.json
├── data/                       # CIFAR-10 auto-downloaded here
└── results/                    # Plots and comparison outputs
```

---

## Setup & Installation

### Google Colab (Recommended)

```python
# Cell 1 — verify GPU
!nvidia-smi

# Cell 2 — install dependencies
!pip install -q thop onnx onnxscript onnxruntime
```

### Local Setup

```bash
# Python 3.10+
pip install torch torchvision
pip install thop onnx onnxruntime onnxscript
```

> For Raspberry Pi deployment, install `onnxruntime` only — no PyTorch required.

```bash
pip install onnxruntime
```

---

## Usage

### 1. Run the Full Pipeline (Notebook)

Open `MobilneNETV2FInal.ipynb` in Colab or Jupyter and run cells in order:

| Cell | Description |
|------|-------------|
| 1    | Config & seed setup |
| 2    | Data loading & augmentation |
| 3    | Utility functions (train/eval/metrics) |
| 4    | Baseline MobileNetV2 training |
| 5    | Structured pruning |
| 6–7  | DARTS operations & search network |
| 8–9  | DARTS architecture search + final training |
| 10   | DARTS + Pruning combined pipeline |
| 11   | Comprehensive model comparison & visualisation |
| 12   | ONNX export (12 variants) |
| 13   | Edge benchmark |

### 2. Export Models to ONNX

```python
# Run the export cell or:
python export_12_models.py --models_dir ./models --output_dir ./onnx_edge_models
```

### 3. Benchmark on Raspberry Pi

Transfer `onnx_edge_models/` to your Pi, then:

```bash
python benchmark_rpi.py --onnx_dir ./onnx_edge_models --n_runs 100
```

### 4. Run Benchmark in Colab

```python
run_benchmark(onnx_dir="./onnx_edge_models", n_warmup=10, n_runs=100)
```

---

## Evaluation Metrics

The benchmark suite reports the following per model:

| Metric | Description |
|--------|-------------|
| **Avg Latency (ms)** | Mean inference time per image |
| **Min Latency (ms)** | Best-case hardware ceiling |
| **Max Latency (ms)** | Worst-case (thermal throttle / OS jitter) |
| **P95 Latency (ms)** | 95th-percentile — real-world worst case |
| **Std Dev (ms)** | Measurement consistency |
| **Throughput (FPS)** | Images per second = 1000 / avg_ms |
| **Speedup (×)** | baseline_lat / model_lat |
| **Latency Reduction (%)** | (1 − model_lat/baseline_lat) × 100 |
| **Model Size (MB)** | ONNX file size on disk |
| **Params (M)** | Parameter count from ONNX graph |
| **FLOPs** | Multiply-accumulate operations (via `thop`) |
| **Top-1 Accuracy (%)** | CIFAR-10 test set accuracy |

---

## ONNX Export & Edge Deployment

Each of the 4 model architectures is exported in 3 ONNX variants (12 total):

| Variant | Description |
|---------|-------------|
| **FP32** | Standard full-precision ONNX export |
| **INT8** | Dynamic quantisation (weights quantised to 8-bit) |
| **Prep** | FP32 graph with ONNX simplification & optimisation |

A `manifest.json` is generated alongside, listing all exported models with their metadata.

**To transfer to Raspberry Pi:**
```bash
# Zip in Colab
import shutil
shutil.make_archive('onnx_edge_models', 'zip', '.', 'onnx_edge_models')

# SCP to Pi
scp onnx_edge_models.zip pi@raspberrypi.local:~/
```

---

## Configuration

All hyperparameters are centralised in the `Config` class:

```python
class Config:
    SEED               = 42
    BATCH_SIZE         = 128
    NUM_CLASSES        = 10

    # Baseline training
    BASELINE_EPOCHS    = 50
    BASELINE_LR        = 0.1

    # Pruning
    PRUNE_AMOUNT       = 0.3        # 30% filter sparsity
    FINETUNE_EPOCHS    = 30

    # DARTS Search
    DARTS_LAYERS           = 6
    DARTS_INIT_CHANNELS    = 16
    DARTS_SEARCH_EPOCHS    = 50
    DARTS_WEIGHT_LR        = 0.025
    DARTS_ARCH_LR          = 3e-4

    # DARTS Final Training
    DARTS_TRAIN_EPOCHS     = 50
    DARTS_TRAIN_LR         = 0.025
```

---

## Results

Four model variants are compared across accuracy, size, and speed:

| Model | Accuracy | Params | Speedup |
|-------|----------|--------|---------|
| Baseline MobileNetV2 | — | ~2.2M | 1.0× (reference) |
| Pruned MobileNetV2 | — | <2.2M | >1.0× |
| DARTS Final | — | ~0.3M | — |
| DARTS + Pruned | — | <0.3M | Highest |

> Exact numbers populate after training. Visualisation plots (accuracy vs. params, inference time, FLOPs) are generated automatically in Cell 11.

---

## Tech Stack

| Category | Tools |
|----------|-------|
| Deep Learning | PyTorch, torchvision |
| NAS | DARTS (custom implementation) |
| Pruning | `torch.nn.utils.prune`, custom structured pruning |
| Model Export | ONNX, onnxruntime, onnxscript |
| FLOPs Counting | `thop` |
| Visualisation | Matplotlib |
| Training Infra | Google Colab (T4 GPU) |
| Edge Runtime | ONNX Runtime (Raspberry Pi) |
| Language | Python 3.10+ |

---

## Acknowledgements

- [MobileNetV2](https://arxiv.org/abs/1801.04381) — Sandler et al., 2018
- [MobileNetV3](https://arxiv.org/abs/1905.02244) — Howard et al., 2019
- [DARTS](https://arxiv.org/abs/1806.09055) — Liu et al., 2019
- [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) — Krizhevsky, 2009
