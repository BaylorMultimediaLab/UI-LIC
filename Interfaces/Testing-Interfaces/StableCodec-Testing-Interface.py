"""
Unified Interface For Learned Image Compression (UI-LIC) - StableCodec Testing Interface

This interface maps UI-LIC configuration parameters to StableCodec's compression entry point (`src/compress.py`).
It manages paths for the core diffusion model (SD-Turbo), auxiliary ELIC checkpoint, model weights, input dataset paths,
and structures reconstructed images and bitstreams into target subdirectories.
"""

import os
from base_interface import BaseInterface


class StableCodecInterface(BaseInterface):
    """
    Testing Interface for StableCodec (One-step diffusion image compression) evaluation.
    """

    TASK_NAME = "StableCodec"
    USE_MODULE_EXECUTION = False
    EXECUTION_PATH = "src/compress.py"

    # -----------------------------
    # ONLY TRUE REQUIRED ARGS
    # -----------------------------
    REQUIRED_ARGS = [
        "sd_path",
        "elic_path",
        "codec_path",
        "test_dataset",
        "save_dir",
    ]

    # -----------------------------
    # OPTIONAL FLAGS ONLY
    # -----------------------------
    ACTION_FLAGS = [
        "color_fix",
        "enable_xformers_memory_efficient_attention",
    ]

    # -----------------------------
    # INPUT CONTRACT
    # -----------------------------
    DEFAULT_VARS = {
        "sd_path": "LIC-Models/StableCodec/sd-turbo",
        "elic_path": "LIC-Models/StableCodec/elic.pth",
        "codec_path": "None",
        "test_dataset": None,
        "save_dir": None,
    }

    # -----------------------------
    # CLI MAPPING
    # -----------------------------
    CLI_MAPPING = {
        "sd_path": "--sd_path",
        "elic_path": "--elic_path",
        "codec_path": "--codec_path",
        "test_dataset": "--img_path", # Mapping test_dataset to --img_path for StableCodec

        # derived automatically
        "rec_path": "--rec_path",
        "bin_path": "--bin_path",

        "color_fix": "--color_fix",
        "enable_xformers_memory_efficient_attention":
            "--enable_xformers_memory_efficient_attention",
    }

    def __init__(self, job_args=None, global_args=None):
        super().__init__(job_args, global_args)

        # -----------------------------
        # PATH ROBUSTNESS
        # -----------------------------
        # Ensure paths are absolute relative to project root before execution CWD changes
        for key in ["sd_path", "elic_path", "codec_path", "test_dataset", "save_dir"]:
            if self.params.get(key):
                self.params[key] = os.path.abspath(os.path.expanduser(self.params[key]))

        # -----------------------------
        # SINGLE DIRECTORY CONTRACT
        # -----------------------------
        save_dir = self.params.get("save_dir")
        if not save_dir:
            raise ValueError("Missing save_dir")

        rec_path = os.path.join(save_dir, "reconstruction")
        bin_path = os.path.join(save_dir, "bitstreams")

        self.params["rec_path"] = rec_path
        self.params["bin_path"] = bin_path

        os.makedirs(rec_path, exist_ok=True)
        os.makedirs(bin_path, exist_ok=True)

        # -----------------------------
        # VALIDATION
        # -----------------------------
        missing = [
            k for k in self.REQUIRED_ARGS
            if not self.params.get(k)
        ]
        if missing:
            raise ValueError(f"Missing required StableCodec args: {missing}")

    def execute(self):
        super().execute()