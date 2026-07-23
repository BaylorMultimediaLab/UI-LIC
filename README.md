# Unified Interface For Learned Image Compression (UI-LIC)

This repository provides a unified framework for managing, training, and testing multiple state-of-the-art Learned Image Compression (LIC) models through a single, consistent interface. It simplifies the complex setup and execution requirements of various research models into a streamlined workflow.

---



## Core Workflow

### 1. Environment Setup
**Note:** This repository targets Linux-based systems with an NVIDIA GPU. We tested primarily on Ubuntu 24.04. If you run into issues on your operating system (especially with the automatic setup and GUI commands), please file an issue on our GitHub page.

#### System Prerequisites
Before setting up your virtual environments, ensure your system has the necessary external libraries and tools installed.

##### Docker (Required for VMAF)
The evaluation pipeline uses **Docker** to calculate the **VMAF** (Video Multi-Method Assessment Fusion) metric. This ensures that a correctly compiled version of FFmpeg with `libvmaf` is available regardless of your host OS configuration.
- **Requirement:** Docker Desktop (Mac/Windows) or Docker Engine (Linux) must be installed and running.
- **Implementation:** The system uses the `mwader/static-ffmpeg` image. Local images are dynamically mounted into the container as read-only volumes for comparison, avoiding the need for a complex local FFmpeg installation.

##### OS-Level Dependencies
For standard frame manipulation and other video-based tasks, the `ffmpeg` binary is still recommended on your host system:

```bash
# For Ubuntu/Debian based systems
sudo apt-get update
sudo apt-get install ffmpeg
```

Install and activate [Anaconda](https://www.anaconda.com/docs/getting-started/miniconda/main)

#### Environment setup
Each LIC model often requires a specific environment with distinct dependencies. 
- **Batch Setup (`quick-start.py`, recommended!) :** Use this to automatically create environments and download weights for ALL integrated models at once.
  ```bash
  python quick-start.py
  ```
  **Features:**
  - **Interactive Selection:** Choose specifically which models to set up and which pretrained weights to download.
  - **Quality/Lambda Selection:** For models like StableCodec or HPCM, you can select specific quality levels to save bandwidth and disk space.
  - **Confirmation Summary:** Review a full plan (including checks for previously installed environments or weights) before any changes are made.
  - **Automatic Dependency Mapping:** Uses the recommended Python versions and requirements files defined for each model automatically.
- **Individual Setup (`create-env.py`):** Use this to create an environment for a single model with custom settings.
  ```bash
  python create-env.py
  ```

#### Qualities & Weights Explanation

Different models use different scales for their pretrained weights. Generally:
- **Lambda (λ):** A higher λ value means the model is optimized for higher quality (and higher bitrate), while a lower λ means higher compression (and lower quality).
- **Metric:** Models are typically optimized for either **MSE** (standard PSNR-focused) or **MS-SSIM** (perceptual-focused).

##### Model-Specific Scales:
- **StableCodec:** Uses `ft` (finetuned) numbers. Higher numbers (e.g., `ft32`) target extreme compression (~0.005 bpp), while lower numbers (e.g., `ft2`) provide higher quality (~0.035 bpp).
- **RwkvCompress (LALIC):** Uses quality levels `q1` to `q6`. `q1` is the highest compression (lowest bitrate), and `q6` is the highest quality.
- **HPCM:** Provides **Base** and **Large** versions. Each has 6 quality levels for both MSE and MS-SSIM metrics.
- **LIC-TCM:** Provides `N=128` (Large) and `N=64` (Small) variants. The quality ranges from λ=0.0025 (highest compression) to λ=0.05 (highest quality).

---
### 2. GUI Evaluation (inference and analysis)
Launch the GUI app with `python ./GUI-Visualizer/desktop_app.py`.

From the start page, you can select the inference dataset directory, enable/disable codecs for inference and analyis, tune codec-specific settings, and launch the evaluation pipeline.
<img width="2644" height="1433" alt="image" src="https://github.com/user-attachments/assets/4da8d15a-1be4-478b-aff9-f2f35a55e1b4" />

After inference has been performed, on the "Visual Comparison" tab you can compare two image reconstructions side by side. The vertical blue line is a slider to adjust the viewport of the two images. The quality metrics may be toggled, and a number of error overlay maps be visualized. For example, this image shows the LPIPS feature maps, conveying the areas of greatest perceptual error.

<img width="3276" height="1531" alt="image" src="https://github.com/user-attachments/assets/68625281-425a-41f5-a405-086042620818" />

On the "Metrics Report" tab, you can view all the quantitative metrics for each codec and input image individually, or you can view the mean results of each codec.

<img width="2644" height="1433" alt="image" src="https://github.com/user-attachments/assets/c80d3634-e42f-4aff-aac4-5b8ee423c515" />



---
### 3. CLI Configuration Generation (`configure-jobs.py`)
Instead of manually editing JSON files, use `configure-jobs.py` to interactively build your queue for CLI-based training and inference jobs.

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

### 4. Running the Dispatcher (`dispatcher.py`)
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

## Integrated Models

The following models are integrated into the platform, each with a specialized interface to bridge their unique CLI requirements:

### **StableCodec**
*Taming One-Step Diffusion for Extreme Image Compression (ICCV 2025)*
- **Recommended Python Version:** 3.10
- **Core Concept:** Uses a one-step diffusion process (SD-Turbo) combined with a dual-branch coding structure.
- **Strength:** Exceptional visual realism at ultra-low bitrates (as low as 0.005 bpp).

#### 50-Series Compatibility Changes
- **`StableCodec.py`:** Added `_encode_latent` helper that casts inputs to the model's native dtype before VAE encoding; denoised latents are also cast back before decoding to prevent dtype mismatches on 50-series cards.
- **`latent_codec.py`:** `get_mask_four_parts` now accepts and propagates an explicit `dtype` argument for all four checkerboard masks; `compress_group_with_mask` and `decompress_group_with_mask` cast their return tensors back to the input dtype.

---

### **ELIC**
*Efficient Learned Image Compression with Context-Adaptive Masked Modeling*
- **Recommended Python Version:** 3.10
- **Strength:** State-of-the-art Rate-Distortion performance, balancing complexity and compression.

#### 50-Series Compatibility Changes
- **`Network.py`:** Imports of `trunc_normal_`, `GaussianConditional`, and `ste_round` now use try/except fallback chains to support both old and new `timm`/`compressai` API locations.
- **`Network.py`:** `load_state_dict` now calls `self.update(force=True)` after loading weights to rebuild entropy coding tables required by newer compressai versions.
- **`requirements.txt`:** Loosened from pinned CUDA 11 packages to `>=`-bounded versions targeting PyTorch 2.4+; added `numpy<2.0.0` upper bound to avoid ABI breakage.

---

### **DCVC-RT**
*Deep Contextual Video Compression - Real Time*
- **Recommended Python Version:** 3.12
- **Strength:** Optimized for low-latency and real-time performance. Foundation for video tasks.

#### 50-Series Compatibility Changes
- **`rans.cpp`:** Output buffer allocation increased to `4× symbol count + 10000` with a pointer bounds assertion, fixing a buffer-overrun crash on sm_120 (RTX 50-series) hardware.
- **`setup.py`:** Removed the `-arch=native` nvcc flag; compute capability is now detected at runtime instead of being baked in at build time.
- **`requirements.txt`:** Added `ninja` and `nvidia-cuda-nvcc` to ensure the JIT/AOT build toolchain is always available.
- **Testing interface:** Extracted a `compile_extensions` method so `create-env.py` can trigger C++ and CUDA extension compilation automatically after pip install.

---

### **LIC-TCM**
*Learned Image Compression with Mixed Transformer-CNN Architectures*
- **Recommended Python Version:** 3.10
- **Strength:** Superior context modeling for complex textures using Transformers.

#### 50-Series Compatibility Changes
- **`requirements.txt`:** CUDA index URL updated from `cu121` to `cu128`; PyTorch/torchvision/torchaudio minimums bumped to `>=2.4.0`/`>=0.19.0`/`>=2.4.0` for sm_120 (Blackwell) wheel support.
- **Testing interface:** Added explicit `ENV_PATH` and `WORKING_DIR` class attributes so the dispatcher and `create-env.py` can locate the environment without relying on inferred paths.

---

### **LIC-HPCM**
*Learned Image Compression with Hierarchical Progressive Context Modeling*
- **Recommended Python Version:** 3.8
- **Strength:** Optimized for hardware acceleration and fast parallel decoding.

#### 50-Series Compatibility Changes
- **`HPCM_Base`, `HPCM_Base_PhiContext`, `HPCM_Large`:** `adaptive_params_list` parameters no longer hard-code `device='cuda'` at construction, removing the import-time CUDA dependency.
- **`entropy_models/__init__.py` and `entropy_models.py`:** Added try/except relative/absolute import fallbacks for the compiled `_CXX` extension.
- **`requirements.txt`:** PyTorch/torchvision/torchaudio minimums bumped to `>=2.4.0`/`>=0.19.0`/`>=2.4.0`.
- **Testing interface:** Added `compile_extensions` and `_check_and_install_dependencies` to automatically build the `unbounded_rans` C++ extension if it is missing from the active environment.

---

### **RwkvCompress**
*Efficient Learned Image Compression via RWKV architecture*
- **Recommended Python Version:** 3.10
- **Strength:** Global dependency modeling with linear-attention computational efficiency.

#### 50-Series Compatibility Changes
- **`biwkv4_cuda_new.cu`:** Replaced deprecated `k.type()` with `k.scalar_type()` in both `AT_DISPATCH_FLOATING_TYPES` calls to fix a compile error with modern PyTorch.
- **`lalic.py`:** JIT load logic now detects the GPU's compute capability at startup and generates matching `-gencode arch=compute_XY,code=sm_XY` flags dynamically instead of hardcoding `sm_86`; the loaded module is also cached to avoid redundant recompilation.
- **`eval.py`:** Removed erroneous `.item()` call on IQA metric tensors.
- **`requirements.txt`:** Added `ninja` and `nvidia-cuda-nvcc` as explicit dependencies.

