import os
import subprocess
from base_interface import BaseInterface


class HPCMTestInterface(BaseInterface):

    TASK_NAME = "HPCM"

    USE_MODULE_EXECUTION = False
    EXECUTION_PATH = "test.py"  # your HPCM eval script

    REQUIRED_ARGS = ["model_name", "checkpoint", "dataset", "save_dir"]

    ACTION_FLAGS = []

    # model name can be HPCM_Base or HPCM_Large
    DEFAULT_VARS = {
        "model_name": "HPCM_Base",
        "checkpoint": None,
        "dataset": None,
        "num": 60,

        # root output dir (contains reconstructed/ + latents/)
        "save_dir": "./outputs"
    }

    # -------------------------
    # Aliases (unified naming style)
    # -------------------------
    ALIASES = {
        "m": "model_name",
        "model": "model_name",

        "ckpt": "checkpoint",

        "data": "dataset",
        "test_dataset": "dataset",
        "input": "dataset",

        "out": "save_dir",
        "output_dir": "save_dir",
    }

    # -------------------------
    # CLI mapping (matches HPCM script)
    # -------------------------
    CLI_MAPPING = {
        "model_name": "--model_name",
        "checkpoint": "--checkpoint",
        "dataset": "--dataset",
        "num": "-num",
        "save_dir": "--save_dir",
    }

    def __init__(self, job_args=None, global_args=None):
        super().__init__(job_args, global_args)

        if not self.params.get("dataset"):
            raise ValueError("dataset is required")

        if not self.params.get("checkpoint"):
            raise ValueError("checkpoint is required")

        if self.params.get("save_dir"):
            os.makedirs(self.params["save_dir"], exist_ok=True)

    def execute(self):
        """
        Ensure HPCM eval compatibility before execution.
        """

        # -------------------------
        # checkpoint validation
        # -------------------------
        ckpt = self.params.get("checkpoint")

        if ckpt is None:
            raise ValueError("checkpoint is required")

        # allow single or list, but normalize to string (HPCM uses single ckpt loop externally)
        if isinstance(ckpt, list):
            if len(ckpt) == 1:
                self.params["checkpoint"] = ckpt[0]
            else:
                # optional: allow multi-ckpt runs by passing first one or raise
                self.params["checkpoint"] = ckpt

        # -------------------------
        # ensure save_dir exists (root output)
        # -------------------------
        os.makedirs(self.params["save_dir"], exist_ok=True)

        super().execute()