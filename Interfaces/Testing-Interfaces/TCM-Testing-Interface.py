import os
import subprocess
from base_interface import BaseInterface

class LICTCMTestInterface(BaseInterface):

    TASK_NAME = "LIC-TCM-Test"
    
    USE_MODULE_EXECUTION = False
    
    EXECUTION_PATH = "eval.py" 
    
    # Checkpoint and data are mandatory
    REQUIRED_ARGS = ["checkpoint", "data"]
    
    # Flags for argparse 'action="store_true"'
    ACTION_FLAGS = ["cuda", "real"]

    DEFAULT_VARS = {
        "checkpoint": None,
        "data": None,
        "clip_max_norm": 1.0,
        "cuda": True,
        "real": False
    }

    ALIASES = {
        "c": "checkpoint",
        "d": "data",
        "dataset": "data"
    }

    CLI_MAPPING = {
        "checkpoint": "--checkpoint",
        "data": "--data",
        "clip_max_norm": "--clip_max_norm",
        "cuda": "--cuda",
        "real": "--real"
    }

    def _check_and_install_dependencies(self):
        """Checks for TCM evaluation requirements."""
        python_exec = os.path.join(self.ENV_PATH, "bin", "python3") if hasattr(self, 'ENV_PATH') else "python3"
        pkgs = ["pytorch_msssim", "PIL", "torch", "torchvision"]
        
        for pkg in pkgs:
            try:
                subprocess.check_call([python_exec, "-c", f"import {pkg}"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except subprocess.CalledProcessError:
                print(f" -> [WARNING] Missing dependency: {pkg}")

    def execute(self):
        self._check_and_install_dependencies()
        super().execute()