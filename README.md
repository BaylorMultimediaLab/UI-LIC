# Unified Interface For Learned Image Compression (LIC)

This repository provides a unified framework for managing, training, and testing multiple state-of-the-art Learned Image Compression (LIC) models through a single, consistent interface. It simplifies the complex setup and execution requirements of various research models into a streamlined workflow.

---

## Getting Started

### 1. Environment Setup (`create-env.py`)
Each LIC model often requires a specific environment with distinct dependencies. The `create-env.py` script automates the creation of these Conda environments.

**How to use:**
```bash
python create-env.py
```
When prompted:
- **Environment path:** Enter the directory where you want the environment created (e.g., `StableCodec/training-testing-env`).
- **Requirements path:** Point to the model's `requirements.txt` (e.g., `StableCodec/requirements.txt`).
- **Python version:** Specify the version (defaults to 3.10 if left blank).

The script will create the environment and install all dependencies via pip automatically.

### 2. Configuration (`arguments.json`)
The `arguments.json` file is the central brain of the platform. It allows you to define global settings and queue multiple tasks across different models.

- **`global_arguments`**: Common parameters shared across all tasks (e.g., `learning_rate`, `batch_size`, `patch_size`, `cuda`).
- **`tasks`**: A dictionary of specific jobs to run.
  - `task_name`: Matches the registered interface name (e.g., `StableCodec`, `ELIC`).
  - `directory`: The root folder of the model code.
  - `env_path`: Relative path to the Conda environment to use for that task.
  - `arguments`: Task-specific overrides or additions.

### 3. Running the Dispatcher (`dispatcher.py`)
The `dispatcher.py` script parses your configuration and executes the tasks in sequence. It handles environment switching, directory management, and command construction automatically.

**How to use:**
```bash
# Run the training queue defined in arguments.json
python dispatcher.py --train

# Use a custom configuration file
python dispatcher.py --train --args_json my_config.json
```

---

## Integrated Models

The following models are integrated into the platform, each with a specialized interface to bridge their unique CLI requirements:

### **StableCodec**
*Taming One-Step Diffusion for Extreme Image Compression (ICCV 2025)*
- **Recommended Python Version:** 3.10
- **Core Concept:** Uses a one-step diffusion process (SD-Turbo) combined with a dual-branch coding structure.
- **Strength:** Exceptional visual realism at ultra-low bitrates (as low as 0.005 bpp) by leveraging generative priors.
- **Workflow:** Typically involves a base training stage followed by GAN-based finetuning.
- **Additional Files:** Requires installation of stable diffusion turbo https://huggingface.co/stabilityai/sd-turbo.

### **ELIC**
*Efficient Learned Image Compression with Context-Adaptive Masked Modeling*
- **Recommended Python Version:** 3.10
- **Core Concept:** A high-performance model utilizing unevenly masked modeling and spatial-channel context.
- **Strength:** Highly efficient coding with state-of-the-art Rate-Distortion performance, balancing complexity and compression.

### **DCVC-RT**
*Deep Contextual Video Compression - Real Time*
- **Recommended Python Version:** 3.12
- **Core Concept:** Optimized for low-latency and real-time performance.
- **Strength:** Excellent for scenarios requiring fast inference and high-quality I-frame (Intra) compression as a foundation for video tasks.

### **LIC-TCM**
*Learned Image Compression with Mixed Transformer-CNN Architectures*
- **Recommended Python Version:** 3.10
- **Core Concept:** Employs Transformers to capture long-range dependencies in the latent space.
- **Strength:** Superior context modeling compared to traditional CNN-based approaches, leading to better compression of complex textures.

### **LIC-HPCM**
*Learned Image Compression with Hierarchical Progressive Context Modeling*
- **Recommended Python Version:** 3.8
- **Core Concept:** Combines different context modeling strategies to maximize parallelization during decoding.
- **Strength:** Optimized for hardware acceleration and fast decoding without sacrificing significant compression efficiency.
- **Required Actions:** Compile LIC-HPCM/src/entropy_models/entropy_coders/unbounded_rans through ./setup.sh
  Compile LIC-HPCM/src/entropy_models through python setup.py build_ext --inplace
- **Implementation Decisions:** Choose between two models [HPCM_Base/HPCM_Large]

### **JPEG-AI**
- **Recommended Python Version:** 3.8
- **Core Concept:** The standardized approach for the next generation of AI-based image coding.
- **Strength:** Focused on interoperability and rigorous testing across a wide range of content and bitrates, following the JPEG-AI Common Test Conditions (CTC).

---

## Project Structure
- `Interfaces/`: Contains the logic for translating unified arguments to model-specific CLI flags.
- `base_interface.py`: The core class that handles command building and environment execution.
- `dispatcher.py`: The entry point for running multi-task queues.
- `[ModelName]/`: Original research codebases integrated as sub-modules.
