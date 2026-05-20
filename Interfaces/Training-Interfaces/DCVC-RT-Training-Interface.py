from base_interface import BaseInterface

class DCVCRTTrainInterface(BaseInterface):

    TASK_NAME = "DCVC-RT"
    
    # Executing the script directly since it uses standard python __main__ entry point
    USE_MODULE_EXECUTION = False
    
    EXECUTION_PATH = "Training/train_dcvc_rt_intra.py"
    
    # 'train_root' is the only explicitly required argparse value in the script
    REQUIRED_ARGS = ["train_root"]
    
    # DCVC-RT does NOT use action switches (even device is passed explicitly as --device cuda)
    ACTION_FLAGS = []

    # Mirrored exactly from the _parse_args() defaults of the DCVC-RT training script
    DEFAULT_VARS = {
        "train_root": None,
        "val_root": None,
        "out_dir": "Training/runs/intra",
        "patch_size": 256,
        "batch_size": 8,
        "num_workers": 2,
        "lr": 1e-4,
        "lambda_rd": 0.01,
        "max_steps": 20000,
        "qp": -1,
        "device": "cuda",
        "seed": 0,
        "log_every": 50,
        "save_every": 1000,
        "val_every": 1000,
        "val_batches": 25
    }

    # Map the unified standard names from your JSON to DCVC-RT's specific variable names
    ALIASES = {
        "train_dataset": "train_root",
        "test_dataset": "val_root",
        "save_path": "out_dir",
        "learning_rate": "lr",
        "lambda": "lambda_rd",
        "lmbda": "lambda_rd",
        "bs": "batch_size"
    }

    # Map the internal configuration standard names to the argparse CLI flags
    CLI_MAPPING = {
        "train_root": "--train_root",
        "val_root": "--val_root",
        "out_dir": "--out_dir",
        "patch_size": "--patch_size",
        "batch_size": "--batch_size",
        "num_workers": "--num_workers",
        "lr": "--lr",
        "lambda_rd": "--lambda_rd",
        "max_steps": "--max_steps",
        "qp": "--qp",
        "device": "--device",
        "seed": "--seed",
        "log_every": "--log_every",
        "save_every": "--save_every",
        "val_every": "--val_every",
        "val_batches": "--val_batches"
    }
    
    def __init__(self, job_args=None, global_args=None):
        # Call the parent BaseInterface init to load and merge all arguments
        super().__init__(job_args, global_args)
        
        # --- UNIFIED TRANSLATION LOGIC ---
        
        # 1. Translate the unified 'cuda' boolean into DCVC-RT's 'device' string
        if "cuda" in self.params:
            if self.params["cuda"] is True:
                self.params["device"] = "cuda"
            elif self.params["cuda"] is False:
                self.params["device"] = "cpu"
            # Remove 'cuda' so the builder doesn't try to append it
            del self.params["cuda"]
            
        # 2. Translate unified list patch sizes (e.g., [256, 256]) to a single integer
        if "patch_size" in self.params and isinstance(self.params["patch_size"], (list, tuple)):
            # Grab the first dimension to satisfy DCVC-RT's single-int argparse requirement
            self.params["patch_size"] = self.params["patch_size"][0]