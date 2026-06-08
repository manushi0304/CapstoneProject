# Edge AI Optimization Pipeline
### Neural Architecture Search and Model Compression for Resource-Constrained Hardware

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Deploying deep learning models on edge devices without sacrificing performance**

A unified optimization pipeline combining Differentiable Architecture Search (DARTS) and L1-norm structural pruning to bridge the gap between theoretical model compression and practical hardware deployment on resource-constrained IoT devices.

---

## 🎯 Overview

Edge AI deployment faces a critical challenge: standard deep learning models designed for GPU-accelerated servers struggle on resource-constrained hardware like Raspberry Pi, resulting in latency that's 20x slower than server-grade performance. This project addresses this fundamental problem.

### The Problem

Training a deep learning model on a T4 GPU achieves **0.17ms** inference time. Deploy that same model on a Raspberry Pi 4, and inference time balloons to **4ms** or more—making real-time applications impossible. The issue isn't the hardware—it's that models aren't designed for these constraints.

### Our Solution

An automated optimization pipeline that:
- **Searches** for optimal architectures using DARTS (Differentiable Architecture Search)
- **Compresses** models through L1-norm structural pruning
- **Deploys** optimized networks on ARM-based edge hardware with validated performance

---

## ✨ Key Features

- **🔍 Neural Architecture Search**: DARTS-based automatic architecture discovery
- **✂️ Intelligent Pruning**: L1-norm structural pruning for parameter reduction
- **⚡ Hardware-Aware**: Optimized specifically for ARM Cortex-A72 CPUs (Raspberry Pi 4)
- **📊 Multi-Model Support**: MobileNetV2, MobileNetV3, and EfficientNet baselines
- **🎓 Research-Grade**: Comprehensive benchmarking and reproducible experiments

---

## 🚀 Results

Experimental validation on CIFAR-10 demonstrates dramatic improvements:

| Metric | Achievement |
|--------|-------------|
| **Parameter Compression** | **99%+** reduction while maintaining accuracy |
| **Hardware Speedup** | Up to **6.68x** faster inference on Raspberry Pi 4 |
| **Inference Latency** | **Sub-millisecond** inference times for mobile CNNs |
| **Memory Efficiency** | Overcomes severe CPU memory limitations |
| **Architecture Discovery** | Automated optimal configuration learning |

### Performance Highlights

- ✅ Preserved predictive confidence on compressed models
- ✅ Tightened class boundaries for visually similar objects
- ✅ Real-time inference without dedicated neural processing units
- ✅ Colossal latency reductions through architecture rebuilding

---

## 🏗️ Architecture

### Optimization Pipeline

```
Input Model (MobileNetV2/V3/EfficientNet)
           ↓
    ┌──────────────────┐
    │  DARTS Search    │  ← Continuous architecture optimization
    │  (Cloud-based)   │
    └──────────────────┘
           ↓
    ┌──────────────────┐
    │  L1-Norm Pruning │  ← Structural parameter compression
    └──────────────────┘
           ↓
    ┌──────────────────┐
    │  Hardware Deploy │  ← Raspberry Pi 4 benchmarking
    │  (Edge Device)   │
    └──────────────────┘
           ↓
    Optimized Edge Model
```

### Two-Phase Approach

1. **Cloud-Based Theoretical Optimization**
   - DARTS architecture search
   - L1-norm structural pruning
   - Model validation and refinement

2. **Physical Deployment & Benchmarking**
   - Raspberry Pi 4 deployment
   - Hardware-intensive performance testing
   - Real-world latency measurement

---

## 🛠️ Technical Stack

### Software Environment
- **Framework**: PyTorch 1.9+
- **Python**: 3.8+
- **Architecture Search**: DARTS (Differentiable Architecture Search)
- **Pruning Method**: L1-norm structural pruning
- **Dataset**: CIFAR-10

### Hardware Environment
- **Development**: Cloud GPU (for architecture search)
- **Deployment Target**: Raspberry Pi 4 (ARM Cortex-A72 CPU)
- **Constraints**: No dedicated GPU, limited RAM

### Baseline Models
- MobileNetV2
- MobileNetV3-Small
- EfficientNet

---

## 📦 Installation

### Prerequisites
```bash
# Python 3.8 or higher
python --version

# PyTorch with CPU support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Clone and Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/edge-ai-optimization.git
cd edge-ai-optimization

# Install dependencies
pip install -r requirements.txt

# (Optional) Setup for Raspberry Pi
# See docs/raspberry-pi-setup.md for detailed instructions
```

---

## 🚦 Quick Start

### 1. Architecture Search (Cloud/GPU)
```bash
# Run DARTS search on baseline model
python search.py \
    --model mobilenetv2 \
    --dataset cifar10 \
    --epochs 50 \
    --gpu 0
```

### 2. Model Pruning
```bash
# Apply L1-norm structural pruning
python prune.py \
    --model searched_architecture.pth \
    --prune-ratio 0.99 \
    --output pruned_model.pth
```

### 3. Edge Deployment
```bash
# Deploy to Raspberry Pi and benchmark
python deploy.py \
    --model pruned_model.pth \
    --device cpu \
    --benchmark
```

### 4. Inference
```bash
# Run inference on edge device
python inference.py \
    --model pruned_model.pth \
    --input test_image.jpg
```

---

## 📊 Benchmarking

### Performance Comparison

Run comprehensive benchmarks on your edge device:

```bash
python benchmark.py \
    --models mobilenetv2,mobilenetv3,efficientnet \
    --iterations 1000 \
    --warmup 100
```

### Key Metrics Tracked
- **Inference Latency** (ms)
- **Memory Usage** (MB)
- **CPU Utilization** (%)
- **Model Accuracy** (%)
- **Parameter Count**

---

## 🔬 Methodology

### DARTS (Differentiable Architecture Search)

Instead of manually designing network architectures, DARTS treats architecture search as a continuous optimization problem:

- **Automatic Operation Selection**: Learns whether to use 3×3 convolutions, depthwise separable convolutions, skip connections, etc.
- **Gradient-Based**: Uses backpropagation to optimize architecture parameters
- **Efficient**: Dramatically faster than reinforcement learning-based NAS methods

### L1-Norm Structural Pruning

- **Layer-wise Importance**: Identifies and removes unimportant filters/channels
- **Structured Removal**: Maintains model structure for efficient inference
- **Iterative Refinement**: Gradual pruning with fine-tuning steps

---

## 📚 Research Highlights

### Novel Contributions

1. **Unified Pipeline**: First work to combine DARTS and structural pruning for edge deployment
2. **Hardware Validation**: Extensive real-world benchmarking on Raspberry Pi 4
3. **Multi-Objective Optimization**: Balances accuracy, latency, and memory simultaneously
4. **Quantization Analysis**: Demonstrates INT8 quantization limitations on ARM CPUs without hardware accelerators

### Experimental Insights

- **99%+ Compression**: Achieves extreme parameter reduction while maintaining performance
- **6.68x Speedup**: Dramatic hardware acceleration through architecture optimization
- **Sub-millisecond Inference**: Enables real-time applications on edge devices
- **No NPU Required**: Proves feasibility without dedicated neural processing units

---

## 🎯 Use Cases

### Ideal Applications

- 🤖 **Autonomous Devices**: Drones, robots with onboard decision-making
- 📹 **Smart Cameras**: Real-time object detection and classification
- 🏠 **IoT Sensors**: Edge-based anomaly detection
- 📱 **Mobile Apps**: On-device ML without cloud dependency
- 🚗 **Automotive**: Driver assistance systems

### When to Use This Pipeline

✅ Deploying CNNs on resource-constrained hardware  
✅ Need for real-time inference (<10ms)  
✅ Limited memory and power budgets  
✅ Cannot rely on cloud connectivity  
✅ Want to avoid expensive NPU/TPU hardware  


## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## 🙏 Acknowledgments

This research project builds upon foundational work in:
- DARTS (Liu et al., 2019)
- MobileNet architectures (Sandler et al., Howard et al.)
- EfficientNet (Tan & Le, 2019)
- Neural Architecture Search literature

Special thanks to the open-source ML community and edge computing researchers.


## 🔮 Future Work

- [ ] Extend to object detection (YOLO, SSD)
- [ ] Support for semantic segmentation models
- [ ] AutoML integration for hyperparameter tuning
- [ ] Mixed-precision quantization strategies
- [ ] Multi-device deployment orchestration
- [ ] Energy consumption profiling
- [ ] Federated learning compatibility

---

<div align="center">

