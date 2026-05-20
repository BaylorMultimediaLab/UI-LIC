from base_interface import BaseInterface
import os
import glob
import subprocess
import tempfile
import atexit

class JPEG_AI_TrainInterface(BaseInterface):

    TASK_NAME = "JPEG-AI"
    
    # Executing the script directly via python train.py
    USE_MODULE_EXECUTION = False
    
    # Points directly to the orchestration script file
    EXECUTION_PATH = "scripts/acc_train_scripts/acc_train_local.py"
    
    # Required keys according to train parameters and json loaders
    REQUIRED_ARGS = [
        "data_dir", 
        "val_data_dir", 
        "test_data_dir", 
        "train_url"
    ]
    
    # Flags parsed by the runtime without trailing true/false string requirements
    ACTION_FLAGS = [
        "overfit", 
        "copy_to_train_url_dir"
    ]

    DEFAULT_VARS = {
        "data_dir": None,
        "lst": "",
        "val_data_dir": None,
        "val_lst": "",
        "batch_size": 8,
        "seed": 42,
        "test_data_dir": None,
        "N": 128,
        "N_UV": 64,
        "hyper_decoder_type": "basic",
        "hyper_scale_decoder_type": "hsd",
        "mse_weight": 1.0,
        "use_automatic_testing": False,
        "automatic_testing_epoch_period": 1,
        "enable_gvae": False,
        "amp": False,
        "sigma_quant_level": 1,
        "sigma_quant_max": 1.0,
        "sigma_quant_min": 0.001,
        "cube_flag_thre": 0.5,
        "loss_weights": "1.0",
        "l1": False,
        "opt_type": "adam",
        "skip_thre": 0.0,
        "rec_dir": "rec",
        "cfg_path": [],
        "vae_encoder_type_list": [],
        "vae_decoder_type_list": [],
        "frozen_part": "",
        "overfit": False,
        "resume_from_stage": "",
        "train_url": "experiments/interactive_run_01/output",
        "train_cfg_json": None,
        "train_stages_json": None,
        "generate_test_summary": False,
        "use_automatic_testing_best": False,
        "automatic_resume_on_crash": False,
        "copy_to_train_url_dir": None
    }

    ALIASES = {
        "d": "data_dir",
        "vd": "val_data_dir",
        "td": "test_data_dir",
        "bs": "batch_size",
        "out": "train_url",
        "cfg": "cfg_path",
        "train_config": "train_cfg_json",
        "stages_config": "train_stages_json",
        "train_dataset": "data_dir",
        "test_dataset": "test_data_dir",
        "output_directory" : "train_url"
    }

    CLI_MAPPING = {
        "data_dir": "--data_dir",
        "lst": "--lst",
        "val_data_dir": "--val_data_dir",
        "val_lst": "--val_lst",
        "batch_size": "--batch_size",
        "seed": "--seed",
        "test_data_dir": "--test_data_dir",
        "N": "--N",
        "N_UV": "--N_UV",
        "hyper_decoder_type": "--hyper_decoder_type",
        "hyper_scale_decoder_type": "--hyper_scale_decoder_type",
        "mse_weight": "--mse_weight",
        "use_automatic_testing": "--use_automatic_testing",
        "automatic_testing_epoch_period": "--automatic_testing_epoch_period",
        "enable_gvae": "--enable_gvae",
        "amp": "--amp",
        "sigma_quant_level": "--sigma_quant_level",
        "sigma_quant_max": "--sigma_quant_max",
        "sigma_quant_min": "--sigma_quant_min",
        "cube_flag_thre": "--cube_flag_thre",
        "loss_weights": "--loss_weights",
        "l1": "--l1",
        "opt_type": "--opt_type",
        "skip_thre": "--skip_thre",
        "rec_dir": "--rec_dir",
        "cfg_path": "--cfg_path",
        "vae_encoder_type_list": "--vae_encoder_type_list",
        "vae_decoder_type_list": "--vae_decoder_type_list",
        "frozen_part": "--frozen_part",
        "overfit": "--overfit",
        "resume_from_stage": "--resume_from_stage",
        "train_url": "--train_url",
        "train_cfg_json": "--train_cfg_json",
        "train_stages_json": "--train_stages_json",
        "generate_test_summary": "--generate_test_summary",
        "use_automatic_testing_best": "--use_automatic_testing_best",
        "automatic_resume_on_crash": "--automatic_resume_on_crash",
        "copy_to_train_url_dir": "--copy_to_train_url_dir"
    }

    def _prepare_list_files(self):
        """1. Auto-populates 'lst'/'val_lst'. 2. Writes to temp files."""
        extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
        for key, dir_key in [("lst", "data_dir"), ("val_lst", "val_data_dir")]:
            if not self.params.get(key) and self.params.get(dir_key):
                files = []
                for ext in extensions:
                    files.extend(glob.glob(os.path.join(self.params[dir_key], ext)))
                self.params[key] = ",".join(files)
            
            content = self.params.get(key, "")
            if isinstance(content, str) and "," in content:
                tmp = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt')
                tmp.write(content.replace(',', '\n'))
                tmp.close()
                self.params[key] = tmp.name
                atexit.register(lambda path=tmp.name: os.remove(path) if os.path.exists(path) else None)
                print(f"[{self.TASK_NAME}] Generated manifest for {key}: {tmp.name}")

    def __init__(self, job_args=None, global_args=None):
        super().__init__(job_args, global_args)
        self._prepare_list_files()
        
        # Aggressive Parameter Sanitization
        cleaned_params = {}
        for key, val in self.params.items():
            # Skip None, empty lists, empty strings, and empty dicts
            if val is None or val == [] or val == "" or val == {}:
                continue
            
            # Convert Boolean (True/False) to Integer (1/0)
            if isinstance(val, bool):
                cleaned_params[key] = 1 if val else 0
            else:
                cleaned_params[key] = val
        
        self.params = cleaned_params
        self._check_and_compile_backend()

    def _check_and_compile_backend(self):
        task_root = "JPEG-AI"
        destination_dir = os.path.join(task_root, "src/codec/entropy_coding/lib_wrappers/mans")
        compiled_extensions = glob.glob(os.path.join(destination_dir, "ans*.so"))
        
        if compiled_extensions:
            found_binary = os.path.basename(compiled_extensions[0])
            print(f"[{self.TASK_NAME}] Verification complete: Found binary ({found_binary}).")
        else:
            print(f"\n[WARNING] {self.TASK_NAME} is missing compiled extensions.")
            user_choice = input("Would you like to compile the arithmetic coder now for JPEG-AI? [y/N]: ").strip().lower()
            if user_choice in ['y', 'yes']:
                target_env = os.path.abspath(os.path.expanduser(self.params.get("env_path", "")))
                print(f"Compiling: conda run --prefix {target_env} make build_test_libs")
                try:
                    subprocess.run(["conda", "run", "--prefix", target_env, "make", "build_test_libs"], 
                                   cwd=task_root, check=True)
                    print("[SUCCESS] Compiled successfully!")
                except subprocess.CalledProcessError as e:
                    print(f"[ERROR] Compilation failed: {e.returncode}.")
            else:
                print("Skipping compilation.")