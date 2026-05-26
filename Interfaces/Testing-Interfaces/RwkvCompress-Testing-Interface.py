import os
import subprocess
from base_interface import BaseInterface


class RwkvCompressTestInterface(BaseInterface):

    TASK_NAME = "RwkvCompress"

    USE_MODULE_EXECUTION = False
    EXECUTION_PATH = "eval.py"

    REQUIRED_ARGS = ["model", "checkpoints", "qualities", "input_dir", "save_dir"]

    ACTION_FLAGS = ["cuda", "half", "real", "verbose"]

    DEFAULT_VARS = {
        "model": "LALIC",
        "entropy_coder": "ans",
        "cuda": True,
        "half": False,
        "real": True,
        "verbose": True,
        "checkpoints": [],
        "qualities": [],
        "input_dir": None,
        "save_dir": None,
        "result": "result.json"
    }

    # -------------------------
    # Aliases (expanded for consistency)
    # -------------------------
    ALIASES = {
        "m": "model",
        "model_name": "model",

        "ckpt": "checkpoints",
        "checkpoint": "checkpoints",

        "q": "qualities",
        "quality": "qualities",

        "dataset": "input_dir",
        "test_dataset": "input_dir",
        "input": "input_dir",

        "output_directory": "save_dir",
    }

    # -------------------------
    # CLI mapping matches eval.py exactly
    # -------------------------
    CLI_MAPPING = {
        "model": "-m",
        "entropy_coder": "-c",
        "cuda": "--cuda",
        "half": "--half",
        "real": "--real",
        "verbose": "-v",

        "checkpoints": "-p",
        "qualities": "-q",

        "input_dir": "-i",
        "save_dir": "-s",
        "result": "-r",
    }

    def __init__(self, job_args=None, global_args=None):
        super().__init__(job_args, global_args)

        # normalize cuda flag if passed in alternative form
        if "cuda" in self.params:
            if self.params["cuda"] is True:
                self.params["cuda"] = True
            elif self.params["cuda"] is False:
                self.params["cuda"] = False

    def execute(self):
        """
        Ensures list args (checkpoints/qualities) are properly formatted
        before execution.
        """
        # Ensure checkpoints + qualities are lists (critical for argparse -p/-q nargs="*")
        if isinstance(self.params.get("checkpoints"), str):
            self.params["checkpoints"] = [self.params["checkpoints"]]

        if isinstance(self.params.get("qualities"), str):
            self.params["qualities"] = [self.params["qualities"]]

        if not isinstance(self.params.get("checkpoints"), list):
            raise ValueError("checkpoints must be a list")

        if not isinstance(self.params.get("qualities"), list):
            raise ValueError("qualities must be a list")

        super().execute()