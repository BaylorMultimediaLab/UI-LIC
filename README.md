# Unified Interface For Learned Image Compression (LIC)

This repository provides a unified framework for managing, training, and testing multiple state-of-the-art Learned Image Compression (LIC) models through a single, consistent interface. It simplifies the complex setup and execution requirements of various research models into a streamlined workflow.

---

## Core Workflow

### 1. Environment Setup (`create-env.py` & `quick-start-env.py`)
Each LIC model often requires a specific environment with distinct dependencies. 

- **Individual Setup (`create-env.py`):** Use this to create an environment for a single model with custom settings.
  ```bash
  python create-env.py
  ```
- **Batch Setup (`quick-start-env.py`, recommended!) :** Use this to automatically create environments for ALL integrated models at once in a specified directory.
  ```bash
  python quick-start-env.py [optional base_path]
  ```
  This script uses the recommended Python versions and requirements files defined for each model automatically.

### 2. Configuration Generation (`configure-jobs.py`)
Instead of manually editing JSON files, use `configure-jobs.py` to interactively build your job queue.

**What it does:**
- Scans `Interfaces/` to find all registered models (e.g., StableCodec, ELIC).
- Prompts for **Global Arguments** (shared across all tasks like `cuda`, `batch_size`).
- Prompts for **Task-Specific Arguments** for each model you want to run.
- Automatically handles argument aliases and provides default values.
- Generates a valid `arguments.json` file ready for the dispatcher.

**How to use:**
```bash
# Start interactive configuration
python configure-jobs.py

# Options:
# --train       : Configure training jobs
# --test        : Configure testing/evaluation jobs
# --output      : Specify custom output filename (default: arguments.json)
```

### 3. Running the Dispatcher (`dispatcher.py`)
The `dispatcher.py` script is the execution engine that processes the `arguments.json` queue.

**What it does:**
- **Environment Switching:** Automatically runs each task within its dedicated Conda environment.
- **Path Validation:** Interactively verifies that all input datasets and checkpoints exist before starting.
- **Safety Checks:** Verifies that dataset images meet the minimum `patch_size` requirements to prevent PyTorch dataloader crashes.
- **Automated Evaluation:** After testing tasks finish, it automatically triggers `evaluation.py` to calculate final metrics and aggregate results.

**How to use:**
```bash
# Execute the training and/or testing queue
python dispatcher.py --train --test

# Optional: Specify a custom configuration file
python dispatcher.py --train --args_json my_config.json
```

---

## Technical Architecture

### Unified Interfaces (`Interfaces/`)
The `Interfaces/` directory contains the "bridge" logic for each model. Each interface file defines:
- `TASK_NAME`: The identifier used in the JSON config.
- `CLI_MAPPING`: Maps unified argument names to the model's specific CLI flags.
- `REQUIRED_ARGS`: Ensures the dispatcher doesn't start a job with missing parameters.
- `ALIASES`: Allows flexible naming (e.g., `data`, `dataset`, and `test_dataset` all map to the same parameter).

### Evaluation Pipeline (`evaluation.py`)
The dispatcher automatically hands off results to `evaluation.py`. This script handles:
- Calculating PSNR, SSIM, LPIPS, and BPP.
- **VMAF Evaluation:** Optional high-quality perceptual metric calculated via Dockerized FFmpeg.
- Aggregating results into structured reports in the `save-dir`.

**Enabling VMAF:**
To enable VMAF, add `"use_vmaf": true` to the `evaluation` block in your `arguments.json`, or pass the `--use_vmaf` flag when running `evaluation.py` manually.

---

## System Prerequisites
Before setting up your virtual environments, ensure your system has the necessary external libraries and tools installed.

### Docker (Required for VMAF)
The evaluation pipeline uses **Docker** to calculate the **VMAF** (Video Multi-Method Assessment Fusion) metric. This ensures that a correctly compiled version of FFmpeg with `libvmaf` is available regardless of your host OS configuration.
- **Requirement:** Docker Desktop (Mac/Windows) or Docker Engine (Linux) must be installed and running.
- **Implementation:** The system uses the `mwader/static-ffmpeg` image. Local images are dynamically mounted into the container as read-only volumes for comparison, avoiding the need for a complex local FFmpeg installation.

### OS-Level Dependencies
For standard frame manipulation and other video-based tasks, the `ffmpeg` binary is still recommended on your host system:

```bash
# For Ubuntu/Debian based systems
sudo apt-get update
sudo apt-get install ffmpeg
```

---

## Integrated Models

The following models are integrated into the platform, each with a specialized interface to bridge their unique CLI requirements:

### **StableCodec**
*Taming One-Step Diffusion for Extreme Image Compression (ICCV 2025)*
- **Recommended Python Version:** 3.10
- **Core Concept:** Uses a one-step diffusion process (SD-Turbo) combined with a dual-branch coding structure.
- **Strength:** Exceptional visual realism at ultra-low bitrates (as low as 0.005 bpp).

### **ELIC**
*Efficient Learned Image Compression with Context-Adaptive Masked Modeling*
- **Recommended Python Version:** 3.10
- **Strength:** State-of-the-art Rate-Distortion performance, balancing complexity and compression.

### **DCVC-RT**
*Deep Contextual Video Compression - Real Time*
- **Recommended Python Version:** 3.12
- **Strength:** Optimized for low-latency and real-time performance. Foundation for video tasks.

### **LIC-TCM**
*Learned Image Compression with Mixed Transformer-CNN Architectures*
- **Recommended Python Version:** 3.10
- **Strength:** Superior context modeling for complex textures using Transformers.

### **LIC-HPCM**
*Learned Image Compression with Hierarchical Progressive Context Modeling*
- **Recommended Python Version:** 3.8
- **Strength:** Optimized for hardware acceleration and fast parallel decoding.

### **RwkvCompress**
*Efficient Learned Image Compression via RWKV architecture*
- **Recommended Python Version:** 3.10
- **Strength:** Global dependency modeling with linear-attention computational efficiency.

