<div align="center">

# ⚡ VortexGrid-Engine

### Production Distributed LLM Training Platform

A production-grade distributed training framework for large language models featuring **PyTorch FSDP**, **DeepSpeed ZeRO**, **Tensor Parallelism**, **Elastic Ray Launchers**, **Kubernetes-native orchestration**, **fault-tolerant checkpointing**, and **enterprise-scale experiment tracking**.

<br>

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-Enabled-76B900?logo=nvidia&logoColor=white)
![DeepSpeed](https://img.shields.io/badge/DeepSpeed-ZeRO-blue)
![FSDP](https://img.shields.io/badge/FSDP-Distributed-orange)
![Ray](https://img.shields.io/badge/Ray-Distributed-028CF0?logo=ray)
![Kubernetes](https://img.shields.io/badge/Kubernetes-Native-326CE5?logo=kubernetes&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-Apache--2.0-blue)

</div>

---

# Overview

Training modern Large Language Models requires far more than distributing gradients across multiple GPUs. Production training systems must efficiently manage parameter sharding, communication overhead, checkpoint reliability, experiment reproducibility, and cluster orchestration while maintaining high hardware utilization.

**VortexGrid-Engine** is a production-oriented distributed LLM training platform that unifies **FSDP**, **DeepSpeed ZeRO**, **Tensor Parallelism**, **mixed precision**, **elastic distributed execution**, **fault-tolerant checkpointing**, and **real-time telemetry** into a modular training infrastructure.

Rather than serving as a collection of training scripts, the project explores how modern large-scale AI training systems are architected—from distributed execution and memory optimization to experiment tracking, monitoring, and Kubernetes-native deployment.

---

# ✨ Features

### 🚀 Distributed Training

- Distributed Data Parallel (DDP)
- Fully Sharded Data Parallel (FSDP)
- DeepSpeed ZeRO-1 / ZeRO-2 / ZeRO-3
- Tensor Parallelism
- Elastic multi-node execution

### ⚡ Memory Optimization

- Mixed Precision Training
- Gradient Checkpointing
- Activation Recomputation
- Optimizer State Sharding
- Checkpoint Sharding

### 📊 Experimentation

- TensorBoard Integration
- Weights & Biases Tracking
- Hyperparameter Optimization
- Automated Experiment Logging
- Training Benchmarking

### 🏭 Production Infrastructure

- Ray Distributed Launcher
- Kubernetes-native Deployment
- Dockerized Runtime
- Fault Recovery
- Auto Resume
- Production Telemetry

---

# Why VortexGrid?

Modern foundation models are trained using sophisticated distributed infrastructures that combine parameter sharding, efficient communication, resilient checkpointing, and production observability. Reproducing these capabilities requires much more than implementing a training loop.

VortexGrid explores these engineering challenges through a unified platform that combines:

- Distributed training strategies
- Large-scale experiment management
- Fault-tolerant execution
- Production telemetry
- Cloud-native deployment
- Performance optimization

The result is a modular training framework designed for scalable AI systems while remaining reproducible, extensible, and production-focused.

---

# 📈 Performance Highlights

| Metric | Target |
|---------|-------:|
| GPU Memory Reduction | **>65%** |
| Training Strategy | **DDP • FSDP • ZeRO-1/2/3** |
| Precision | **FP32 • FP16 • BF16** |
| Distributed Execution | **Multi-GPU / Multi-Node** |
| Fault Recovery | **Automatic Resume** |

### Core Engineering Components

| Component | Implementation |
|-----------|----------------|
| **Distributed Runtime** | Torch Distributed, NCCL, Ray |
| **Parallelism** | DDP, FSDP, Tensor Parallelism, ZeRO |
| **Memory Optimization** | Mixed Precision, Gradient Checkpointing, Activation Recomputation |
| **Experiment Platform** | TensorBoard, Weights & Biases, Optuna |
| **Deployment** | Docker, Kubernetes |
| **Observability** | Prometheus, GPU Telemetry, Performance Profiling |

---

## Table of Contents

- System Architecture
- Repository Structure
- Technology Stack
- Installation
- Quick Start
- Distributed Training
- Monitoring
- Roadmap
- Contributing
- License

---

# 🏗️ System Architecture

VortexGrid-Engine separates orchestration, distributed execution, memory optimization, and experiment management into independent layers. This modular architecture enables scalable training across single-GPU development environments and distributed multi-node clusters while maintaining reproducibility and fault tolerance.

```text
                               VortexGrid-Engine
┌──────────────────────────────────────────────────────────────────────────────┐
│                 Production Distributed LLM Training Platform                 │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                              Training Configuration
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                           Training Orchestrator                             │
├──────────────────────────────────────────────────────────────────────────────┤
│ Config Loader │ Launch Manager │ Experiment Manager │ CLI │ Logging │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                     Distributed Training Runtime                            │
├──────────────────────────────────────────────────────────────────────────────┤
│ DDP │ FSDP │ DeepSpeed ZeRO │ Tensor Parallelism │ NCCL │ Torch Distributed │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                     Memory Optimization Engine                              │
├──────────────────────────────────────────────────────────────────────────────┤
│ Mixed Precision │ Gradient Checkpointing │ Activation Recomputation │        │
│ Optimizer Sharding │ Checkpoint Sharding │ CUDA Memory Profiling            │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│               Experiment Tracking & Monitoring                              │
├──────────────────────────────────────────────────────────────────────────────┤
│ TensorBoard │ Weights & Biases │ Prometheus │ GPU Telemetry │ Profiling │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                     Docker & Kubernetes Deployment                          │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

# ⚙️ Core Engineering Concepts

VortexGrid-Engine is designed around production training principles commonly found in modern large-scale AI infrastructure.

---

## Distributed Training

The training runtime supports multiple distributed execution strategies to efficiently scale model training while minimizing communication overhead.

### Supported Strategies

- Distributed Data Parallel (DDP)
- Fully Sharded Data Parallel (FSDP)
- DeepSpeed ZeRO-1
- DeepSpeed ZeRO-2
- DeepSpeed ZeRO-3
- Tensor Parallelism
- Elastic Multi-Node Training

---

## Memory Optimization

Training large language models requires aggressive memory optimization to maximize hardware utilization.

Key capabilities include:

- Mixed Precision (FP16 / BF16)
- Gradient Checkpointing
- Activation Recomputation
- Optimizer State Sharding
- Checkpoint Sharding
- CUDA Memory Profiling

---

## Fault-Tolerant Training

Long-running distributed jobs require resilient checkpointing and recovery mechanisms.

The platform provides:

- Automatic checkpoint sharding
- Asynchronous checkpoint saving
- Auto resume
- Fault recovery
- Optimizer state restoration
- Training state synchronization

---

## Experiment Management

Training experiments are fully reproducible through integrated tracking and configuration management.

Supported capabilities include:

- Hyperparameter tracking
- Automated experiment logging
- Bayesian optimization
- Performance benchmarking
- Training comparison
- Reproducible configuration management

---

## Production Monitoring

Built-in observability enables continuous monitoring of distributed training performance.

Collected metrics include:

- GPU utilization
- Memory consumption
- Training throughput
- Step latency
- Communication overhead
- Loss convergence
- Cluster health

---

# 📂 Repository Structure

```text
VortexGrid-Engine/
├── .github/
│   └── workflows/
│       ├── ci-cd.yaml                   # Linting, type checks, unit tests
│       └── docker-build.yaml            # Multi-arch CUDA Docker container 
├── manifests/
│   ├── k8s/
│   │   ├── base/
│   │   │   ├── custom-metrics-exporter.yaml
│   │   │   ├── prometheus-service.yaml
│   │   │   └── ray-cluster.yaml         # KubeRay CRD operator config
│   │   └── jobs/
│   │       ├── elastic-training-job.yaml # K8s PyTorchJob / RayJob spec
│   │       └── auto-recovery-cron.yaml
│   └── ray/
│       └── cluster-config.yaml      # Multi-node Ray cluster configuration
├── docker/
│   ├── Dockerfile.cuda                  # Slim CUDA + PyTorch runtime image
│   └── docker-compose.telemetry.yaml    # Local Prometheus + Grafana 
├── configs/
│   ├── deepspeed/
│   │   ├── zero1.json                   # ZeRO Stage 1 config
│   │   ├── zero2.json                   # ZeRO Stage 2 config
│   │   └── zero3_offload.json           # ZeRO Stage 3 + CPU Offload config
│   ├── fsdp/
│   │   ├── full_shard.yaml              # PyTorch FSDP FULL_SHARD strategy
│   │   └── hybrid_shard.yaml            # Multi-node Hybrid Sharding spec
│   ├── hpo/
│   │   └── optuna_config.yaml           # Bayesian HPO search space 
│   └── model_configs/
│       ├── llama_7b.yaml                # LLaMA architecture specs
│       └── mistral_7b.yaml              # Mistral architecture specs
├── dashboards/
│   └── grafana_gpu_telemetry.json       # Pre-configured Grafana dashboard 
├── src/
│   └── vortexgrid/
│       ├── engine/                      # Core Distributed Engine
│       │   ├── distributed_context.py   # Process group rank 
│       │   ├── fsdp_wrapper.py          # FSDP auto-wrap rules 
│       │   ├── deepspeed_engine.py      # DeepSpeed engine builder 
│       │   └── tensor_parallel.py      # Column & Row parallel linear layer
│       ├── checkpointing/               # State Sharding & Fault Recovery
│       │   ├── async_sharded_saver.py   # Asynchronous multi-GPU 
│       │   ├── state_loader.py          # Resilient optimizer 
│       │   └── fault_tolerance.py       # Elastic rank auto-resume 
│       ├── telemetry/                   # Profiling & Metrics
│       │   ├── metrics_collector.py     # Custom CUDA runtime metrics 
│       │   ├── prometheus_exporter.py   # HTTP Prometheus exporter 
│       │   └── memory_profiler.py       # Detailed PyTorch CUDA memory 
│       ├── tracker/                     # Experiment Tracking
│       │   ├── wandb_tracker.py         # Weights & Biases logging 
│       │   └── tensorboard_tracker.py   # TensorBoard logging wrapper
│       ├── hpo/                         # Hyperparameter Search
│       │   ├── optuna_runner.py         # Distributed Optuna study executor
│       │   └── search_spaces.py       # Hyperparameter search distributions
│       ├── launcher/                    # Cluster Launchers
│       │   ├── ray_launcher.py          # Ray job submitter & auto-scaler
│       │   └── k8s_elastic.py         # Kubernetes distributed job launcher
│       └── models/                      # Architecture Blocks
│           ├── transformer_blocks.py    # LLaMA/Mistral layers with    
                                                #gradient recomputation
│           └── loss_functions.py        # Fused memory cross-entropy loss
├── scripts/
│   ├── setup_env.sh                    # Local environment bootstrap script
│   ├── benchmark_scaling.py           #Scaling & throughput benchmark suite
│   ├── run_distributed_train.py       # Multi-GPU training pipeline entry
│   └── run_hpo_sweep.py                # Distributed Optuna sweep executor
├── tests/
│   ├── unit/                            # System unit test suite
│   └── integration/                     # Multi-process distributed tests
├── pyproject.toml                       # Build specifications
└── requirements.txt                     # Production requirements
```

---

## Module Overview

| Module | Responsibility |
|---------|----------------|
| **distributed/** | Distributed execution using DDP, FSDP, ZeRO, and Tensor Parallelism |
| **trainer/** | Core training engine, optimization, and mixed precision |
| **checkpointing/** | Fault-tolerant checkpointing and automatic recovery |
| **experiments/** | Experiment tracking, benchmarking, and hyperparameter optimization |
| **telemetry/** | GPU monitoring, Prometheus metrics, and performance profiling |
| **launcher/** | Ray launcher and Kubernetes orchestration |
| **benchmarks/** | Distributed training performance evaluation |
| **tests/** | Unit, integration, and distributed system validation |

---

# 🛠️ Technology Stack

| Category | Technologies |
|-----------|--------------|
| Training Framework | PyTorch |
| Distributed Runtime | Torch Distributed, NCCL, Ray |
| Parallelism | DDP, FSDP, DeepSpeed ZeRO, Tensor Parallelism |
| Optimization | Mixed Precision, Gradient Checkpointing |
| Experiment Tracking | TensorBoard, Weights & Biases, Optuna |
| Deployment | Docker, Kubernetes |
| Monitoring | Prometheus, GPU Telemetry |
| Testing | PyTest, GitHub Actions |

---
# 🚀 Installation

## Requirements

- Python **3.10+**
- CUDA Toolkit
- PyTorch 2.x
- Docker & Docker Compose
- Kubernetes (optional)
- NCCL
- Ray
- CMake **3.20+**

Clone the repository:

```bash
git clone https://github.com/BAHAA00111/VortexGrid-Engine.git

cd VortexGrid-Engine
```

Create and activate a virtual environment:

```bash
python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows
# .venv\Scripts\activate
```

Install dependencies:

```bash
pip install --upgrade pip

pip install -e .
```

---

# ⚙️ Quick Start

## Launch a Distributed Training Job

```bash
python train.py \
    --config configs/fsdp_llama.yaml
```

---

## Launch with Ray

```bash
python scripts/launch_ray_cluster.py

python train.py \
    --launcher ray \
    --config configs/fsdp_llama.yaml
```

---

## Run Distributed Benchmarks

```bash
python benchmarks/benchmark_training.py
```

---

## Execute the Test Suite

```bash
pytest tests -v
```

---

## Start Monitoring

Launch Prometheus and TensorBoard dashboards.

```bash
docker compose up -d
```

---

# 🐳 Deployment

VortexGrid-Engine is designed for production-scale distributed training and supports both local and cloud-native deployments.

Deployment options include:

- Docker
- Docker Compose
- Ray Cluster
- Kubernetes
- Multi-Node Training
- Elastic Distributed Execution

Example:

```bash
kubectl apply -f kubernetes/
```

---

# 📊 Observability

The platform includes integrated monitoring for distributed training workloads.

Available telemetry includes:

- GPU utilization
- GPU memory consumption
- Training throughput
- Tokens per second
- Communication overhead
- Step latency
- Checkpoint duration
- Loss convergence
- Learning rate schedule
- Cluster health

Monitoring stack:

- TensorBoard
- Weights & Biases
- Prometheus
- Custom GPU Telemetry

---

# 📚 Technology References

VortexGrid-Engine draws inspiration from modern distributed AI training ecosystems, including:

- PyTorch Distributed
- DeepSpeed
- Ray
- Kubernetes
- NCCL
- TensorBoard
- Weights & Biases
- Prometheus

---

# 📄 License

Licensed under the **Apache 2.0 License**.
---

<div align="center">

## ⭐ Support the Project

If you find **VortexGrid-Engine** useful, consider giving the repository a **Star**.

Your support helps improve project visibility and encourages continued development.

---

**Built for scalable, resilient, and production-ready distributed LLM training.**

</div>
