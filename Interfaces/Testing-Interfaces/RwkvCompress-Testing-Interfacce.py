import os
import subprocess
from base_interface import BaseInterface

class RwkvCompressTestInterface(BaseInterface):

    TASK_NAME = "RwkvCompress"
    
    USE_MODULE_EXECUTION = False
    EXECUTION_PATH = "eval.py"  # Ensure this matches your eval script filename
    REQUIRED_ARGS = ["model", "checkpoints", "qualities", "input_dir"]
    ACTION_FLAGS = ["cuda", "half", "real", "verbose"]

    DEFAULT_VARS = {
        "model": "bmshj2018-factorized",
        "entropy_coder": "ans",
        "cuda": True,
        "half": False,
        "real": True,
        "verbose": True,
        "checkpoints": [],
        "qualities": [],
        "input_dir": None,
        "output_dir": None,
        "result": "result.json"
    }

    ALIASES = {
        "m": "model",
        "test_dataset": "input_dir"
    }

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
        "output_dir": "-o",
        "result": "-r"
    }

