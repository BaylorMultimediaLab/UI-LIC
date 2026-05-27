import os
from base_interface import BaseInterface

class VVCInterface(BaseInterface):
    TASK_NAME = "VVC"
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
        cmd.extend(["--codec", "VVC"])
        return cmd
