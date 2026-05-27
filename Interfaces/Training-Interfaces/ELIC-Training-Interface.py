from base_interface import BaseInterface
import os

class ELICTrainInterface(BaseInterface):

    TASK_NAME = "ELIC"
    
    USE_MODULE_EXECUTION = False
    EXECUTION_PATH = "train.py"
    
    # Enforce these to avoid accidental default overwrites
    REQUIRED_ARGS = ["dataset", "savepath"]

    # Mirrored exactly from the argparse defaults in ELIC2/train.py
    DEFAULT_VARS = {
        "dataset": None,
        "epochs": 4000,
        "learning_rate": 1e-4,
        "num_workers": 4,
        "lmbda": 0.015,
        "batch_size": 16,
        "test_batch_size": 32,
        "aux_learning_rate": 1e-3,
        "patch_size": [256, 256],
        "cuda": True,
        "save": True,
        "seed": 1926,
        "clip_max_norm": 1.0,
        "checkpoint": None,
        "pretrained": False,
        "gpu_id": "0",
        "savepath": "./checkpoint",
        "N": 192,
        "M": 320
    }

    # Map the short flags to the standard names
    ALIASES = {
        "d": "dataset",
        "train_dataset": "dataset",
        "e": "epochs",
        "lr": "learning_rate",
        "n": "num_workers",
        "lambda": "lmbda",
        "lambda_rate": "lmbda",
        "bs": "batch_size",
        "c": "checkpoint",
        "gpu": "gpu_id",
        "output_directory": "savepath"
    }

    # Map the standard names to the CLI flags
    CLI_MAPPING = {
        "dataset": "--dataset",
        "epochs": "--epochs",
        "learning_rate": "--learning-rate",
        "num_workers": "--num-workers",
        "lmbda": "--lambda",
        "batch_size": "--batch-size",
        "test_batch_size": "--test-batch-size",
        "aux_learning_rate": "--aux-learning-rate",
        "patch_size": "--patch-size",
        "gpu_id": "--gpu-id",
        "cuda": "--cuda",
        "save": "--save",
        "seed": "--seed",
        "clip_max_norm": "--clip_max_norm",
        "checkpoint": "--checkpoint",
        "pretrained": "--pretrained",
        "savepath": "--savepath",
        "N": "--N",
        "M": "--M"
    }

    def __init__(self, job_args=None, global_args=None):
        super().__init__(job_args, global_args)
        
        # Ensure paths are absolute or handled correctly
        if self.params.get("savepath"):
            self.params["savepath"] = os.path.abspath(os.path.expanduser(self.params["savepath"]))
        if self.params.get("dataset"):
            self.params["dataset"] = os.path.abspath(os.path.expanduser(self.params["dataset"]))
