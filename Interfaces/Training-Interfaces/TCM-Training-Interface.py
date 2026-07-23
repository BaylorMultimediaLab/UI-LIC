"""
Unified Interface For Learned Image Compression (UI-LIC) - LIC-TCM Training Interface

This interface maps UI-LIC training configuration parameters to LIC-TCM's training script (`train.py`),
handling Transformer-CNN channel settings (N=64 or N=128), MultiStepLR milestone schedules (`lr_epoch`), and loss options (MSE/MS-SSIM).
"""

from base_interface import BaseInterface


class LICTCMTrainInterface(BaseInterface):
    """
    Training Interface for LIC-TCM (Transformer-CNN image compression) training tasks.
    """

    TASK_NAME = "LIC-TCM"
    
    # Executing the script directly since it uses standard python __main__ entry point
    USE_MODULE_EXECUTION = False
    
    # Points directly to the training script file inside your JPEG-AI folder
    EXECUTION_PATH = "train.py"
    
    # Enforce these to avoid accidental default overwrites
    REQUIRED_ARGS = ["train_dataset", "test_dataset", "save_path"]
    
    # Tells BaseInterface NOT to append "True" or "False" to these flags
    # used for training script which don't use these values
    ACTION_FLAGS = ["cuda", "save", "continue_train"]

    # Mirrored exactly from the parse_args defaults of your TCM training script
    DEFAULT_VARS = {
        "model": "bmshj2018-factorized",
        "dataset": None,  # <-- Clean, explicit "empty" state 
        "train_dataset": "train",
        "test_dataset": "test",        
        "epochs": 50,
        "learning_rate": 1e-4,
        "num_workers": 20,
        "lmbda": 3.0,
        "batch_size": 8,
        "test_batch_size": 8,
        "aux_learning_rate": 1e-3,
        "patch_size": [256, 256],
        "cuda": False,  # Using string representation matching your parser strategy
        "save": True,
        "seed": 100.0,
        "clip_max_norm": 1.0,
        "checkpoint": None,
        "type": "mse",
        "save_path": None,
        "skip_epoch": 0,
        "N": 64,
        "lr_epoch": [45, 48],  # Milestones list for MultiStepLR
        "continue_train": True
    }

    # Map the short/common flags to the standard names
    ALIASES = {
        "m": "model",
        "d": "dataset",
        "e": "epochs",
        "lr": "learning_rate",
        "n": "num_workers",
        "lambda": "lmbda",
        "bs": "batch_size",
        "tbs": "test_batch_size",
        "aux_lr": "aux_learning_rate",
        "c": "checkpoint",
        "loss_type": "type"
    }

    # Map the internal configuration standard names to the argparse CLI flags
    CLI_MAPPING = {
        "model": "--model",
        "dataset": "--dataset",
        "epochs": "--epochs",
        "learning_rate": "--learning-rate",
        "num_workers": "--num-workers",
        "lmbda": "--lambda",
        "train_dataset": "--train_dataset",
        "test_dataset": "--test_dataset",
        "batch_size": "--batch-size",
        "test_batch_size": "--test-batch-size",
        "aux_learning_rate": "--aux-learning-rate",
        "patch_size": "--patch-size",
        "cuda": "--cuda",
        "save": "--save",
        "seed": "--seed",
        "clip_max_norm": "--clip_max_norm",
        "checkpoint": "--checkpoint",
        "type": "--type",
        "save_path": "--save_path",
        "skip_epoch": "--skip_epoch",
        "N": "--N",
        "lr_epoch": "--lr_epoch",
        "continue_train": "--continue_train"
    }