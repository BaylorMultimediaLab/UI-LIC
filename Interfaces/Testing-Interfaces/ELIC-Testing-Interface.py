import os
from base_interface import BaseInterface

class LICELICTestInterface(BaseInterface):

    TASK_NAME = "ELIC"
    
    USE_MODULE_EXECUTION = False
    EXECUTION_PATH = "Inference.py" 
    
    REQUIRED_ARGS = ["checkpoint", "dataset", "save_dir"]
    ACTION_FLAGS = ["cuda", "half", "entropy_estimation"]

    DEFAULT_VARS = {
        "checkpoint": None,
        "dataset": None,
        "save_dir": None,
        "cuda": True,
        "half": False,
        "entropy_estimation": False,
        "entropy_coder": "ans",
        "patch": 256
    }

    ALIASES = {
        "c": "entropy_coder",
        "d": "dataset",
        "test_dataset": "dataset",
        "checkpoint": "checkpoint",
        "p": "checkpoint",
        "output": "save_dir",
        "out": "save_dir"
    }

    CLI_MAPPING = {
        "checkpoint": "--path",
        "dataset": "--dataset",
        "save_dir": "--output_path",
        "entropy_coder": "--entropy-coder",
        "cuda": "--cuda",
        "half": "--half",
        "entropy_estimation": "--entropy-estimation",
        "patch": "--patch"
    }

    def __init__(self, job_args=None, global_args=None):
        super().__init__(job_args, global_args)
        
        # Fallback to global test_dataset if data/dataset isn't provided locally
        if not self.params.get("dataset") and global_args and "test_dataset" in global_args:
            self.params["dataset"] = global_args["test_dataset"]
            
        # Ensure paths are absolute or handled correctly
        for key in ["checkpoint", "dataset", "save_dir"]:
            if self.params.get(key):
                self.params[key] = os.path.abspath(os.path.expanduser(self.params[key]))

    def execute(self):
        super().execute()