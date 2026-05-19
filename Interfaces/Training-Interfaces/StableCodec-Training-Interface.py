from base_interface import BaseInterface
import os

class StableCodecTrainInterface(BaseInterface):

    TASK_NAME = "StableCodec"
    
    USE_MODULE_EXECUTION = False

    
    # Update this if your execution file is named differently
    EXECUTION_PATH = "src/train.py"
    
    
    # Enforce these to avoid accidental default overwrites
    REQUIRED_ARGS = ["train_dataset", "test_dataset", "sd_path", "elic_path", "lambda_rate" ]
    
    
    
    # Tells BaseInterface NOT to append "True" or "False" to these boolean flags.
    # If they exist in the JSON as 'true', they will be passed as empty switches (e.g. --save_val)
    ACTION_FLAGS = [
        "enable_xformers_memory_efficient_attention", 
        "gradient_checkpointing", 
        "allow_tf32", 
        "set_grads_to_none"
    ]


    ## USING MIXED PRECISION BF16 rather than no

    # Derived from the explicit argparse usage inside the provided StableCodec training script
    DEFAULT_VARS = {
        "sd_path": None,
        "gradient_accumulation_steps": 1,
        "mixed_precision": "bf16",
        "report_to": "tensorboard",
        "seed": 100,
        "train_dataset": None,
        "test_dataset": None,
        "train_patch_size": 256,
        "train_batch_size": 8,
        "dataloader_num_workers": 4,
        "output_dir": "experiments/StableCodec",
        "enable_xformers_memory_efficient_attention": False,
        "gradient_checkpointing": False,
        "allow_tf32": False,
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "adam_weight_decay": 1e-2,
        "adam_epsilon": 1e-8,
        "max_train_steps": 100000,
        "lambda_l2": 1.0,
        "lambda_lpips": 1.0,
        "lambda_clip": 1.0,
        "max_grad_norm": 1.0,
        "set_grads_to_none": False,
        "checkpointing_steps": 1000,
        "eval_freq": 1000,
        "save_val": True,
        "save_num": 10
    }

    # Map the unified standard names from your JSON to StableCodec's specific variable names
    ALIASES = {
        "patch_size": "train_patch_size",
        "batch_size": "train_batch_size",
        "num_workers": "dataloader_num_workers",
        "save_path": "output_dir",
        "epochs": "max_train_steps"  # Note: Script uses steps instead of epochs
    }

    # Map the internal configuration standard names to the argparse CLI flags
    CLI_MAPPING = {
        "sd_path": "--sd_path",
        "elic_path":  "--elic_path",  
        "lambda_rate": "--lambda_rate",
        "gradient_accumulation_steps": "--gradient_accumulation_steps",
        "mixed_precision": "--mixed_precision",
        "report_to": "--report_to",
        "seed": "--seed",
        "train_dataset": "--train_dataset",
        "test_dataset": "--test_dataset",
        "train_patch_size": "--train_patch_size",
        "train_batch_size": "--train_batch_size",
        "dataloader_num_workers": "--dataloader_num_workers",
        "output_dir": "--output_dir",
        "enable_xformers_memory_efficient_attention": "--enable_xformers_memory_efficient_attention",
        "gradient_checkpointing": "--gradient_checkpointing",
        "allow_tf32": "--allow_tf32",
        "adam_beta1": "--adam_beta1",
        "adam_beta2": "--adam_beta2",
        "adam_weight_decay": "--adam_weight_decay",
        "adam_epsilon": "--adam_epsilon",
        "max_train_steps": "--max_train_steps",
        "lambda_l2": "--lambda_l2",
        "lambda_lpips": "--lambda_lpips",
        "lambda_clip": "--lambda_clip",
        "max_grad_norm": "--max_grad_norm",
        "set_grads_to_none": "--set_grads_to_none",
        "checkpointing_steps": "--checkpointing_steps",
        "eval_freq": "--eval_freq",
        "save_val": "--save_val",
        "save_num": "--save_num"
    }

    def __init__(self, job_args=None, global_args=None):
        # Call the parent BaseInterface init to load and merge all arguments
        super().__init__(job_args, global_args)
        
        # --- UNIFIED TRANSLATION LOGIC ---
        
        # 1. Strip the master 'cuda' unified parameter since this script automatically 
        #    handles devices via HuggingFace Accelerate and doesn't accept a --cuda flag
        if "cuda" in self.params:
            del self.params["cuda"]
            
        # 2. Translate unified list patch sizes (e.g., [256, 256]) to a single integer
        if "train_patch_size" in self.params:
            val = self.params["train_patch_size"]
            if isinstance(val, (list, tuple)):
                # Grab the first dimension to satisfy the script's single-int assumption
                self.params["train_patch_size"] = val[0]
            elif isinstance(val, str):
                self.params["train_patch_size"] = int(val.split()[0])