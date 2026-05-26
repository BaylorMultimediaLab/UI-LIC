import os
import subprocess
from base_interface import BaseInterface

class LICELICTestInterface(BaseInterface):

    TASK_NAME = "ELIC"
    
    USE_MODULE_EXECUTION = True
    EXECUTION_PATH = "playground.custom-evaluation" 
    
    REQUIRED_ARGS = ["checkpoint", "dataset", "save_dir"]
    ACTION_FLAGS = []

    DEFAULT_VARS = {
        "checkpoint": None,
        "dataset": None,
        "save_dir": None,
        "cuda": True,
        "experiment": "eval_run",
        "test_batch_size": 1,
        "num_workers": 1
    }

    ALIASES = {
        "c": "checkpoint",
        "d": "dataset",
        "data": "dataset",
        "test_dataset": "dataset",
        "output": "save_dir",
        "out": "save_dir"
    }

    # Removed save_dir, added rec_path and bin_path
    CLI_MAPPING = {
        "checkpoint": "--checkpoint",
        "dataset": "--dataset",
        "experiment": "--experiment",
        "test_batch_size": "--test-batch-size",
        "num_workers": "-n",
        "cuda": "--cuda",
        "rec_path": "--rec_path",
        "bin_path": "--bin_path"
    }

    def __init__(self, job_args=None, global_args=None):
        super().__init__(job_args, global_args)
        
        # Fallback to global test_dataset if data/dataset isn't provided locally
        if not self.params.get("dataset") and global_args and "test_dataset" in global_args:
            self.params["dataset"] = global_args["test_dataset"]
            
        # Expand user paths ('~') for Linux shell compatibility
        for key in ["checkpoint", "dataset", "save_dir"]:
            if self.params.get(key):
                self.params[key] = os.path.expanduser(self.params[key])

        # --- SAVE_DIR → REC/BIN SPLIT (DCVC-RT Style) ---
        if self.params.get("save_dir"):
            base = self.params["save_dir"]

            self.params["rec_path"] = os.path.join(base, "reconstruction")
            self.params["bin_path"] = os.path.join(base, "bitstreams")

            os.makedirs(self.params["rec_path"], exist_ok=True)
            os.makedirs(self.params["bin_path"], exist_ok=True)

    def execute(self):
        super().execute()