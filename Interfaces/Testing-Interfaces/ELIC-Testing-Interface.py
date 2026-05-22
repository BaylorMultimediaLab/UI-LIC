import os
import subprocess
from base_interface import BaseInterface

class ELICTestInterface(BaseInterface):

    TASK_NAME = "ELIC"
    
    USE_MODULE_EXECUTION = True 
    EXECUTION_PATH = "playground.test" 
    
    REQUIRED_ARGS = ["experiment", "checkpoint", "dataset"]

    DEFAULT_VARS = {
        "experiment": "evaluation",
        "dataset": None,
        "test-batch-size": 1,
        "num_workers": 8,
        "cuda": True,
        "checkpoint": None
    }

    ALIASES = {
        "exp": "experiment",
        "d": "dataset",
        "c": "checkpoint",
        "bs": "test_batch_size",
        "n": "num_workers",
        "test_dataset": "dataset"
    }

    CLI_MAPPING = {
        "experiment": "--experiment",
        "dataset": "--dataset",
        "test-batch-size": "--test-batch-size",
        "num_workers": "--num-workers",
        "cuda": "--cuda",
        "checkpoint": "--checkpoint"
    }
