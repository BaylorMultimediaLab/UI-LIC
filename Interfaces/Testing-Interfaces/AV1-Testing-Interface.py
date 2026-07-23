"""
Unified Interface For Learned Image Compression (UI-LIC) - AV1 Testing Interface

This interface maps UI-LIC evaluation parameters (QP quality parameter, input/output paths) to the standard video
codec evaluation runner (`Standard-Codecs/eval_standard.py`) for the AV1 codec format.
"""

import os
from base_interface import BaseInterface

class AV1Interface(BaseInterface):
    """
    Testing Interface for AV1 standard image/video compression benchmark comparisons.
    """
    TASK_NAME = "AV1"
    ENV_PATH = "LIC-Models/eval-env"
    WORKING_DIR = "LIC-Models/Standard-Codecs"
    USE_MODULE_EXECUTION = False
    EXECUTION_PATH = "eval_standard.py"

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
        cmd.extend(["--codec", "AV1"])
        return cmd
