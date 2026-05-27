import os
from base_interface import BaseInterface

class AV1Interface(BaseInterface):
    TASK_NAME = "AV1"
    USE_MODULE_EXECUTION = False
    EXECUTION_PATH = "Standard-Codecs/eval_standard.py"

    REQUIRED_ARGS = ["qp", "input_dir", "save_dir"]
    DEFAULT_VARS = {
        "qp": 23,
        "input_dir": None,
        "save_dir": None,
    }

    CLI_MAPPING = {
        "qp": "--qp",
        "input_dir": "--input_dir",
        "save_dir": "--save_dir"
    }

    def build_command(self):
        cmd = super().build_command()
        cmd.extend(["--codec", "AV1"])
        return cmd
