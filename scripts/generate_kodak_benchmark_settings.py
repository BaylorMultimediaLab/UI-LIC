#!/usr/bin/env python3
"""
UI-LIC Kodak Benchmark Runner
Runs all integrated LIC models and standard codecs across their respective quality levels
on the Kodak dataset (24 images), starting with DCVC-RT.
Automatically computes RD curves, BD-rates, per-image variance, and generates paper & README reports.
"""

import os
import sys
import json
import subprocess

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.download_kodak import download_kodak

def build_benchmark_config(dataset_dir, output_root):
    dataset_dir = os.path.abspath(os.path.expanduser(dataset_dir))
    output_root = os.path.abspath(os.path.expanduser(output_root))

    tasks = {}

    # 1. DCVC-RT — all 64 supported QP indices (0-63)
    # The model has get_qp_num()=64 discrete quality levels. Running all of them
    # gives the densest possible RD curve and maximises overlap with other anchors.
    for qp in range(64):
        task_id = f"DCVC-RT_qp{qp}"
        tasks[task_id] = {
            "task_name": "DCVC-RT",
            "env_path": "LIC-Models/DCVC-RT-env",
            "directory": "LIC-Models/DCVC-RT",
            "arguments": {
                "model_path": os.path.abspath("LIC-Models/DCVC-RT/weights/cvpr2025_image.pth.tar"),
                "input": os.path.abspath(dataset_dir),
                "qp": qp,
                "save_dir": os.path.join(output_root, task_id)
            }
        }

    # 2. ELIC
    elic_configs = [
        {"name": "ELIC_0450", "ckpt": "LIC-Models/ELIC/weights/ELIC_0450_ft_3980_Plateau.pth.tar"},
        {"name": "ELIC_0150", "ckpt": "LIC-Models/ELIC/weights/ELIC_0150_ft_3980_Plateau.pth.tar"},
        {"name": "ELIC_0032", "ckpt": "LIC-Models/ELIC/weights/ELIC_0032_ft_3980_Plateau.pth.tar"},
        {"name": "ELIC_0016", "ckpt": "LIC-Models/ELIC/weights/ELIC_0016_ft_3980_Plateau.pth.tar"},
        {"name": "ELIC_0008", "ckpt": "LIC-Models/ELIC/weights/ELIC_0008_ft_3980_Plateau.pth.tar"},
        {"name": "ELIC_0004", "ckpt": "LIC-Models/ELIC/weights/ELIC_0004_ft_3980_Plateau.pth.tar"},
    ]
    for cfg in elic_configs:
        task_id = cfg["name"]
        tasks[task_id] = {
            "task_name": "ELIC",
            "env_path": "LIC-Models/ELIC-env",
            "directory": "LIC-Models/ELIC",
            "arguments": {
                "checkpoint": os.path.abspath(cfg["ckpt"]),
                "dataset": os.path.abspath(dataset_dir),
                "save_dir": os.path.join(output_root, task_id)
            }
        }

    # 3. HPCM (Base & Large; MSE & SSIM variants)
    hpcm_configs = [
        # HPCM Base (MSE)
        {"id": "HPCM_Base_mse_0.0018", "model": "HPCM_Base", "ckpt": "LIC-Models/HPCM/weights/hpcm_base_mse_0.0018.pth"},
        {"id": "HPCM_Base_mse_0.0035", "model": "HPCM_Base", "ckpt": "LIC-Models/HPCM/weights/hpcm_base_mse_0.0035.pth"},
        {"id": "HPCM_Base_mse_0.0067", "model": "HPCM_Base", "ckpt": "LIC-Models/HPCM/weights/hpcm_base_mse_0.0067.pth"},
        {"id": "HPCM_Base_mse_0.013",  "model": "HPCM_Base", "ckpt": "LIC-Models/HPCM/weights/hpcm_base_mse_0.013.pth"},
        {"id": "HPCM_Base_mse_0.025",  "model": "HPCM_Base", "ckpt": "LIC-Models/HPCM/weights/hpcm_base_mse_0.025.pth"},
        {"id": "HPCM_Base_mse_0.0483", "model": "HPCM_Base", "ckpt": "LIC-Models/HPCM/weights/hpcm_base_mse_0.0483.pth"},
        # HPCM Base (SSIM)
        {"id": "HPCM_Base_SSIM_2.4",   "model": "HPCM_Base", "ckpt": "LIC-Models/HPCM/weights/hpcm_base_ssim_2.4.pth"},
        {"id": "HPCM_Base_SSIM_4.58",  "model": "HPCM_Base", "ckpt": "LIC-Models/HPCM/weights/hpcm_base_ssim_4.58.pth"},
        {"id": "HPCM_Base_SSIM_8.73",  "model": "HPCM_Base", "ckpt": "LIC-Models/HPCM/weights/hpcm_base_ssim_8.73.pth"},
        {"id": "HPCM_Base_SSIM_16.64", "model": "HPCM_Base", "ckpt": "LIC-Models/HPCM/weights/hpcm_base_ssim_16.64.pth"},
        {"id": "HPCM_Base_SSIM_31.73", "model": "HPCM_Base", "ckpt": "LIC-Models/HPCM/weights/hpcm_base_ssim_31.73.pth"},
        {"id": "HPCM_Base_SSIM_60.5",  "model": "HPCM_Base", "ckpt": "LIC-Models/HPCM/weights/hpcm_base_ssim_60.5.pth"},
        # HPCM Large (MSE)
        {"id": "HPCM_Large_mse_0.0018", "model": "HPCM_Large", "ckpt": "LIC-Models/HPCM/weights/hpcm_large_mse_0.0018.pth"},
        {"id": "HPCM_Large_mse_0.0035", "model": "HPCM_Large", "ckpt": "LIC-Models/HPCM/weights/hpcm_large_mse_0.0035.pth"},
        {"id": "HPCM_Large_mse_0.0067", "model": "HPCM_Large", "ckpt": "LIC-Models/HPCM/weights/hpcm_large_mse_0.0067.pth"},
        {"id": "HPCM_Large_mse_0.013",  "model": "HPCM_Large", "ckpt": "LIC-Models/HPCM/weights/hpcm_large_mse_0.013.pth"},
        {"id": "HPCM_Large_mse_0.025",  "model": "HPCM_Large", "ckpt": "LIC-Models/HPCM/weights/hpcm_large_mse_0.025.pth"},
        {"id": "HPCM_Large_mse_0.0483", "model": "HPCM_Large", "ckpt": "LIC-Models/HPCM/weights/hpcm_large_mse_0.0483.pth"},
        # HPCM Large (SSIM)
        {"id": "HPCM_Large_SSIM_2.4",   "model": "HPCM_Large", "ckpt": "LIC-Models/HPCM/weights/hpcm_large_ssim_2.4.pth"},
        {"id": "HPCM_Large_SSIM_4.58",  "model": "HPCM_Large", "ckpt": "LIC-Models/HPCM/weights/hpcm_large_ssim_4.58.pth"},
        {"id": "HPCM_Large_SSIM_8.73",  "model": "HPCM_Large", "ckpt": "LIC-Models/HPCM/weights/hpcm_large_ssim_8.73.pth"},
        {"id": "HPCM_Large_SSIM_16.64", "model": "HPCM_Large", "ckpt": "LIC-Models/HPCM/weights/hpcm_large_ssim_16.64.pth"},
        {"id": "HPCM_Large_SSIM_31.73", "model": "HPCM_Large", "ckpt": "LIC-Models/HPCM/weights/hpcm_large_ssim_31.73.pth"},
        {"id": "HPCM_Large_SSIM_60.5",  "model": "HPCM_Large", "ckpt": "LIC-Models/HPCM/weights/hpcm_large_ssim_60.5.pth"},
    ]
    for cfg in hpcm_configs:
        task_id = cfg["id"]
        tasks[task_id] = {
            "task_name": "HPCM",
            "env_path": "LIC-Models/HPCM-env",
            "directory": "LIC-Models/HPCM",
            "arguments": {
                "model_name": cfg["model"],
                "checkpoint": os.path.abspath(cfg["ckpt"]),
                "dataset": os.path.abspath(dataset_dir),
                "save_dir": os.path.join(output_root, task_id)
            }
        }

    # 4. RwkvCompress (LALIC q1-q6)
    rwkv_configs = [
        {"id": "RwkvCompress_q1", "ckpt": "LIC-Models/RwkvCompress/weights/lalic-q1.pth"},
        {"id": "RwkvCompress_q2", "ckpt": "LIC-Models/RwkvCompress/weights/lalic-q2.pth"},
        {"id": "RwkvCompress_q3", "ckpt": "LIC-Models/RwkvCompress/weights/lalic-q3.pth"},
        {"id": "RwkvCompress_q4", "ckpt": "LIC-Models/RwkvCompress/weights/lalic-q4.pth"},
        {"id": "RwkvCompress_q5", "ckpt": "LIC-Models/RwkvCompress/weights/lalic-q5.pth"},
        {"id": "RwkvCompress_q6", "ckpt": "LIC-Models/RwkvCompress/weights/lalic-q6.pth"},
    ]
    for cfg in rwkv_configs:
        task_id = cfg["id"]
        tasks[task_id] = {
            "task_name": "RwkvCompress",
            "env_path": "LIC-Models/RwkvCompress-env",
            "directory": "LIC-Models/RwkvCompress",
            "arguments": {
                "checkpoint": os.path.abspath(cfg["ckpt"]),
                "dataset": os.path.abspath(dataset_dir),
                "save_dir": os.path.join(output_root, task_id)
            }
        }

    # 5. LIC-TCM
    tcm_configs = [
        {"id": "LIC-TCM_0.0025",    "ckpt": "LIC-Models/LIC-TCM/weights/tcm_mse_64_0.0025.pth"},
        {"id": "LIC-TCM_0.0035",    "ckpt": "LIC-Models/LIC-TCM/weights/tcm_mse_64_0.0035.pth"},
        {"id": "LIC-TCM_0.0067",    "ckpt": "LIC-Models/LIC-TCM/weights/tcm_mse_64_0.0067.pth"},
        {"id": "LIC-TCM_0.013",     "ckpt": "LIC-Models/LIC-TCM/weights/tcm_mse_64_0.013.pth"},
        {"id": "LIC-TCM_0.025",     "ckpt": "LIC-Models/LIC-TCM/weights/tcm_mse_64_0.025.pth"},
        {"id": "LIC-TCM_64_0.05",   "ckpt": "LIC-Models/LIC-TCM/weights/tcm_mse_64_0.05.pth"},
        {"id": "LIC-TCM_128_0.05",  "ckpt": "LIC-Models/LIC-TCM/weights/tcm_mse_128_0.05.pth"},
    ]
    for cfg in tcm_configs:
        task_id = cfg["id"]
        tasks[task_id] = {
            "task_name": "LIC-TCM",
            "env_path": "LIC-Models/LIC-TCM-env",
            "directory": "LIC-Models/LIC-TCM",
            "arguments": {
                "checkpoint": os.path.abspath(cfg["ckpt"]),
                "dataset": os.path.abspath(dataset_dir),
                "save_dir": os.path.join(output_root, task_id)
            }
        }

    # 6. StableCodec (base, ft2 to ft32)
    stable_configs = [
        {"id": "StableCodec_base", "ckpt": "LIC-Models/StableCodec/weights/stablecodec_base.pkl"},
        {"id": "StableCodec_ft2",  "ckpt": "LIC-Models/StableCodec/weights/stablecodec_ft2.pkl"},
        {"id": "StableCodec_ft3",  "ckpt": "LIC-Models/StableCodec/weights/stablecodec_ft3.pkl"},
        {"id": "StableCodec_ft4",  "ckpt": "LIC-Models/StableCodec/weights/stablecodec_ft4.pkl"},
        {"id": "StableCodec_ft6",  "ckpt": "LIC-Models/StableCodec/weights/stablecodec_ft6.pkl"},
        {"id": "StableCodec_ft8",  "ckpt": "LIC-Models/StableCodec/weights/stablecodec_ft8.pkl"},
        {"id": "StableCodec_ft12", "ckpt": "LIC-Models/StableCodec/weights/stablecodec_ft12.pkl"},
        {"id": "StableCodec_ft16", "ckpt": "LIC-Models/StableCodec/weights/stablecodec_ft16.pkl"},
        {"id": "StableCodec_ft24", "ckpt": "LIC-Models/StableCodec/weights/stablecodec_ft24.pkl"},
        {"id": "StableCodec_ft32", "ckpt": "LIC-Models/StableCodec/weights/stablecodec_ft32.pkl"},
    ]
    for cfg in stable_configs:
        task_id = cfg["id"]
        tasks[task_id] = {
            "task_name": "StableCodec",
            "env_path": "LIC-Models/StableCodec-env",
            "directory": "LIC-Models/StableCodec",
            "arguments": {
                "checkpoint": os.path.abspath(cfg["ckpt"]),
                "dataset": os.path.abspath(dataset_dir),
                "sd_path": os.path.abspath("LIC-Models/StableCodec/sd-turbo"),
                "elic_path": os.path.abspath("LIC-Models/StableCodec/weights/elic.pth"),
                "save_dir": os.path.join(output_root, task_id)
            }
        }

    # 7. Standard Codecs (AV1, HEVC, AVC)
    # - AVC/HEVC spec max QP is 51 (QPs 0–51 = 52 points each).
    # - AV1 max quantizer index is 63 (QPs 0–63 = 64 points).
    # Full coverage from lossless/near-lossless (QP 0) down to max compression.
    for codec in ["AV1", "HEVC", "AVC"]:
        max_qp = 63 if codec == "AV1" else 51
        for qp in range(0, max_qp + 1):
            task_id = f"{codec}_qp{qp}"
            tasks[task_id] = {
                "task_name": codec,
                "env_path": "LIC-Models/eval-env",
                "directory": "LIC-Models/Standard-Codecs",
                "arguments": {
                    "qp": qp,
                    "input_dir": dataset_dir,
                    "save_dir": os.path.join(output_root, task_id)
                }
            }

    eval_tasks = []
    for task_id, info in tasks.items():
        eval_tasks.append({
            "task_name": task_id,
            "save_dir": info["arguments"]["save_dir"],
            "input_dir": dataset_dir
        })

    config_payload = {
        "testing": {
            "global_arguments": {
                "cuda": True,
                "gpu_id": 0,
                "test_dataset": dataset_dir
            },
            "tasks": tasks,
            "evaluation": {
                "env_path": "LIC-Models/eval-env",
                "use_vmaf": True,
                "tasks": eval_tasks
            }
        }
    }

    config_path = os.path.abspath("arguments_kodak_benchmark.json")
    with open(config_path, "w") as f:
        json.dump(config_payload, f, indent=4)

    print(f"Generated benchmark task queue: {config_path} ({len(tasks)} runs total)")
    return config_path

def main():
    dataset_dir = "./data/kodak"
    output_root = "./results/kodak_benchmark"

    print("=== UI-LIC Kodak Multi-Model Benchmark Suite ===")
    print("1. Verifying Kodak Dataset...")
    download_kodak(dataset_dir)

    print("\n2. Building Benchmark Queue (Starting with DCVC-RT)...")
    config_path = build_benchmark_config(dataset_dir, output_root)

    print("\n3. Ready to run Dispatcher!")
    print(f"Run command:\n  python dispatcher.py --test --args_json {config_path}")

if __name__ == "__main__":
    main()
