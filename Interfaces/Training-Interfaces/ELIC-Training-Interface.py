from base_interface import BaseInterface

class ELICTrainInterface(BaseInterface):

    TASK_NAME = "ELIC"
    
    ## USING MODULE EXECUTION RATHER THAN EXPLICIT PATH BASED
    USE_MODULE_EXECUTION = True
    
    EXECUTION_PATH = "playground.train"
    
    # Enforce these to avoid accidental default overwrites
    REQUIRED_ARGS = ["experiment", "dataset", "train_dataset", "test_dataset"]

    # Mirrored exactly from the argparse defaults
    DEFAULT_VARS = {
        "experiment": "0483mse",
        "dataset": None,  
        "train_dataset": None, 
        "test_dataset": None,   
        "train_split": "", 
        "test_split": "",       
        "epochs": 60000,
        "learning_rate": 1e-4,
        "num_workers": 8,
        "lmbda": 0.0483,
        "metrics": "mse",
        "batch_size": 8,
        "test_batch_size": 1,
        "aux_learning_rate": 1e-3,
        "patch_size": [128, 128],
        "gpu_id": 0,
        "cuda": True,
        "save": True,
        "seed": 192.1,
        "clip_max_norm": 1.0,
        "checkpoint": None
    }

    # Map the short flags to the standard names
    ALIASES = {
        "exp": "experiment",
        "d": "dataset",
        "e": "epochs",
        "lr": "learning_rate",
        "n": "num_workers",
        "lambda": "lmbda",
        "bs": "batch_size",
        "c": "checkpoint",
        "gpu": "gpu_id",
        "output_directory": "experiment"
    }

    # Map the standard names to the CLI flags
    CLI_MAPPING = {
        "experiment": "--experiment",
        "dataset": "--dataset",
        "train_dataset": "--train_dataset", 
        "test_dataset": "--test_dataset",   
        "train_split": "--train_split", 
        "test_split": "--test_split",
        "epochs": "--epochs",
        "learning_rate": "--learning-rate",
        "num_workers": "--num-workers",
        "lmbda": "--lambda",
        "metrics": "--metrics",
        "batch_size": "--batch-size",
        "test_batch_size": "--test-batch-size",
        "aux_learning_rate": "--aux-learning-rate",
        "patch_size": "--patch-size",
        "gpu_id": "--gpu_id",
        "cuda": "--cuda",
        "save": "--save",
        "seed": "--seed",
        "clip_max_norm": "--clip_max_norm",
        "checkpoint": "--checkpoint"
    }