import os
import subprocess
from base_interface import BaseInterface

class LICTCMTestInterface(BaseInterface):

    TASK_NAME = "LIC-TCM"
    
    ENV_PATH = "LIC-Models/LIC-TCM-env"
    WORKING_DIR = "LIC-Models/LIC-TCM"
    USE_MODULE_EXECUTION = False
    
    # Fixed typo here
    EXECUTION_PATH = "custom-evaluation.py" 
    
    # Checkpoint, data, and save_dir are mandatory for saving functionality
    REQUIRED_ARGS = ["checkpoint", "data", "save_dir"]
    
    # Flags for argparse 'action="store_true"'
    ACTION_FLAGS = ["cuda", "real"]

    DEFAULT_VARS = {
        "checkpoint": None,
        "data": None,
        "save_dir": None,
        "clip_max_norm": 1.0,
        "cuda": True,
        "real": True,  # Must be True to trigger actual compression/bitstream generation
        "N": 128,
        "M": 320
    }

    # Added aliases so you can use "output" in your dispatcher config
    ALIASES = {
        "c": "checkpoint",
        "d": "data",
        "dataset": "data",
        "output": "save_dir",
        "out": "save_dir",
        "test_dataset": "data"
    }

    CLI_MAPPING = {
        "checkpoint": "--checkpoint",
        "data": "--data",
        "save_dir": "--save_dir",
        "clip_max_norm": "--clip_max_norm",
        "N": "-N",          # <--- THIS IS CRITICAL
        "M": "-M",          # <--- Added M
        "model": "--model", # <--- Add this if your script uses it
        "cuda": "--cuda",
        "real": "--real"
    }

    def __init__(self, job_args=None, global_args=None):
        super().__init__(job_args, global_args)
        
        # Manually resolve the dataset path if the global config used 'test_dataset'
        if not self.params.get("data") and global_args and "test_dataset" in global_args:
            self.params["data"] = global_args["test_dataset"]
            
        # Expand user paths ('~') to ensure they work in the Linux shell
        for key in ["checkpoint", "data", "save_dir"]:
            if self.params.get(key):
                self.params[key] = os.path.expanduser(self.params[key])

        if self.params.get("save_dir"):
            os.makedirs(self.params["save_dir"], exist_ok=True)

    def execute(self):
        # Ensure output directory exists before running
        if self.params.get("save_dir"):
            os.makedirs(self.params["save_dir"], exist_ok=True)
            
        super().execute()