from base_interface import BaseInterface
import os
import glob
import subprocess

class JPEG_AI_TrainInterface(BaseInterface):
    TASK_NAME = "JPEG-AI"
    
    # We define the module here, but we override get_command to handle the launcher
    USE_MODULE_EXECUTION = True
    EXECUTION_PATH = "src.train.CCS.acc_train.multistages_train.train"
    
    REQUIRED_ARGS = ["data_dir", "val_data_dir", "train_url"]

    DEFAULT_VARS = {
        "data_dir": "~/data/jpeg_ai_data/JPEG-AI-Images/val2017-crops-random",
        "lst": "~/data/jpeg_ai_data/JPEG-AI-Images/val2017_short_cropped_list.txt",
        "val_data_dir": "~/data/jpeg_ai_data/JPEG-AI-Images/kodak-crops-random",
        "val_lst": "~/data/jpeg_ai_data/JPEG-AI-Images/kodak_cropped_list.txt",
        "batch_size": 8,
        "workers": 4,
        "beta_list": 0.075,
        "lr": 0.001,
        "base_warmup_epoch": 1.0,
        "anneal_final_lr": 0.0001,
        "lr_type": "warmup_anneal.step",
        "amp": True,
        "cfg_path": ["tools_off.json", "profiles/base.json"],
        "vae_encoder_type_list": ["bop", "hop"],
        "vae_decoder_type_list": ["bop", "hop", "sop"],
        "print_freq": 10,
        "epochs": 10,
        "train_url": "~/data/jpeg_ai_data/output_dir/beta_0.075"
    }

    CLI_MAPPING = {k: f"--{k}" for k in DEFAULT_VARS.keys()}

    def get_command(self):
        """Overrides the command construction to include the DDP launcher."""
        # Launcher prefix
        cmd = ["python3", "-m", "torch.distributed.launch", "--nproc_per_node=1", "--master_port=52951"]
        # The module
        cmd += ["-m", self.EXECUTION_PATH]
        
        # Arguments
        for key, val in self.params.items():
            if key in self.CLI_MAPPING:
                if isinstance(val, list):
                    cmd.extend([self.CLI_MAPPING[key]] + [str(v) for v in val])
                else:
                    cmd.extend([self.CLI_MAPPING[key], str(val)])
        return cmd

    def __init__(self, job_args=None, global_args=None):
        super().__init__(job_args, global_args)
        
        # 1. Expand paths
        path_keys = ["data_dir", "lst", "val_data_dir", "val_lst", "train_url"]
        for key in path_keys:
            if self.params.get(key):
                self.params[key] = os.path.abspath(os.path.expanduser(self.params[key]))
        
        # 2. Compilation check
        self._check_compilation()

    def _check_compilation(self):
        task_root = "JPEG-AI"
        dest = os.path.join(task_root, "src/codec/entropy_coding/lib_wrappers/mans")
        if not glob.glob(os.path.join(dest, "ans*.so")):
            print(f"\n[WARNING] {self.TASK_NAME} missing compiled ANS extensions.")