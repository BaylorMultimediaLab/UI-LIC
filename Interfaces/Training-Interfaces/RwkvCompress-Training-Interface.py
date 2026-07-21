"""
Unified Interface For Learned Image Compression (UI-LIC) - RwkvCompress Training Interface

This interface maps UI-LIC training configuration parameters to RwkvCompress's training entry point (`train.py`),
configuring LALIC model variants, epoch counts, learning rate decay schedules, and dataset directories.
"""

from base_interface import BaseInterface

class RwkvTrainInterface(BaseInterface):
    """
    Training Interface for RwkvCompress (RWKV linear attention) training tasks.
    """

    TASK_NAME = "RwkvCompress"
    
    # Executes the training script directly
    USE_MODULE_EXECUTION = False
    
    # Path relative to your unified model directory
    EXECUTION_PATH = "train.py"
    
    REQUIRED_ARGS = ["train_dataset", "test_dataset", "save_path"]
    
    ACTION_FLAGS = ["cuda", "save", "continue_train"]


    # choose default model from choose from 'bmshj2018-factorized', 
    # 'bmshj2018-factorized-relu', 'bmshj2018-hyperprior', 'mbt2018-mean', 
    # 'mbt2018', 'cheng2020-anchor', 'cheng2020-attn', 'bmshj2018-hyperprior-vbr',
    #  'mbt2018-mean-vbr', 'mbt2018-vbr', 'hrtzxf2022-pcc-rec', 'sfu2023-pcc-rec-pointnet',
    #  'sfu2024-pcc-rec-pointnet2-ssg', 'ssf2020'

    DEFAULT_VARS = {
        "model": "LALIC",
        "epochs": 40,
        "learning_rate": 1e-4,
        "num_workers": 20,
        "lmbda": 0.0067,
        "batch_size": 8,
        "test_batch_size": 8,
        "aux_learning_rate": 1e-3,
        "patch_size": [256, 256],
        "cuda": True,
        "save": True,
        "seed": 100.0,
        "clip_max_norm": 1.0,
        "train_dataset": None,
        "test_dataset": None,
        "checkpoint": None,
        "type": "mse",
        "save_path": None,
        "skip_epoch": 0,
        "N": 128,
        "lr_epoch": [36], 
        "continue_train": True
    }

    ALIASES = {
        "m": "model",
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

    CLI_MAPPING = {
        "model": "--model",
        "train_dataset": "--train_dataset",
        "test_dataset": "--test_dataset",
        "epochs": "--epochs",
        "learning_rate": "--learning-rate",
        "num_workers": "--num-workers",
        "lmbda": "--lambda",
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

    def __init__(self, job_args=None, global_args=None):
        super().__init__(job_args, global_args)
        
        # Debugging
        print(f"[DEBUG] RwkvTrainInterface init: params={self.params}")
        
        # Ensure rates are floats
        if "learning_rate" in self.params:
            self.params["learning_rate"] = float(self.params["learning_rate"])
        if "aux_learning_rate" in self.params:
            self.params["aux_learning_rate"] = float(self.params["aux_learning_rate"])
            
        # Ensure N and epochs are ints
        if "N" in self.params:
            self.params["N"] = int(self.params["N"])
        if "epochs" in self.params:
            self.params["epochs"] = int(self.params["epochs"])
            
        print(f"[DEBUG] RwkvTrainInterface after cast: lr={type(self.params['learning_rate'])}, aux_lr={type(self.params['aux_learning_rate'])}")