import os
import subprocess
from base_interface import BaseInterface


class RwkvCompressTestInterface(BaseInterface):

    TASK_NAME = "RwkvCompress"
    ENV_PATH = "LIC-Models/RwkvCompress-env"
    WORKING_DIR = "LIC-Models/RwkvCompress"

    USE_MODULE_EXECUTION = False
    EXECUTION_PATH = "eval.py"

    REQUIRED_ARGS = ["model", "checkpoint", "quality", "input_dir", "save_dir"]

    ACTION_FLAGS = ["cuda", "half", "real", "verbose"]

    DEFAULT_VARS = {
        "model": "LALIC",
        "entropy_coder": "ans",
        "cuda": True,
        "half": False,
        "real": True,
        "verbose": True,
        "checkpoint": None,
        "quality": "1",
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

        "ckpt": "checkpoint",
        "checkpoints": "checkpoint",

        "q": "quality",
        "qualities": "quality",

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

        "checkpoint": "-p",
        "quality": "-q",

        "input_dir": "-i",
        "save_dir": "-s",
        "rec_path": "-s",
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
        # Ensure checkpoint + quality are lists for the CLI (argparse -p/-q nargs="*")
        if "checkpoint" in self.params:
            ckpt = self.params["checkpoint"]
            if not isinstance(ckpt, list):
                self.params["checkpoint"] = [ckpt]

        if "quality" in self.params:
            q = self.params["quality"]
            if not isinstance(q, list):
                self.params["quality"] = [q]

        super().execute()