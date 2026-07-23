"""
Unified Interface For Learned Image Compression (UI-LIC) - HEVC (H.265) Testing Interface

This interface maps UI-LIC evaluation parameters (QP quality parameter, input/output paths) to the standard video
codec evaluation runner (`Standard-Codecs/eval_standard.py`) for the H.265/HEVC codec format.
"""

import os
from base_interface import BaseInterface

class HEVCInterface(BaseInterface):
    """
    Testing Interface for H.265/HEVC standard image/video compression benchmark comparisons.
    """
    TASK_NAME = "HEVC"
    USE_MODULE_EXECUTION = False
    EXECUTION_PATH = "Standard-Codecs/eval_standard.py"

    REQUIRED_ARGS = ["qp", "input_dir", "save_dir"]
    ACTION_FLAGS = ["use_gpu"]
    DEFAULT_VARS = {
        "qp": 23,
        "use_gpu": False,
        "input_dir": None,
        "save_dir": None,
    }

    CLI_MAPPING = {
        "qp": "--qp",
        "use_gpu": "--use_gpu",
        "input_dir": "--input_dir",
        "save_dir": "--save_dir"
    }

    def build_command(self):
        cmd = super().build_command()
        cmd.extend(["--codec", "HEVC"])
        return cmd
