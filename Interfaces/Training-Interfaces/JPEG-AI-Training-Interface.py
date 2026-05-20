from base_interface import BaseInterface

import os
import glob
import subprocess


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

    # Mirrored defaults extracted directly from common_parameters constructions
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

    # Short flag conversions matching the common parameters architecture
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

    # Complete parameter tracking translation map to CLI execution layer
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

    def __init__(self, job_args=None, global_args=None):
        super().__init__(job_args, global_args)

        # --- INTEGER BOOLEAN CONVERSION ---
        for key, val in self.params.items():
            if isinstance(val, bool):
                self.params[key] = 1 if val else 0
        
        # Remove empty strings, empty lists, and None so they don't create dangling flags
        empty_keys = [k for k, v in self.params.items() if v in ["", [], None]]
        for k in empty_keys:
            self.params.pop(k, None)


        # --- RECOMMENDED JPEG-AI C++ COMPILATION ---
        
        
        task_root = "JPEG-AI"
        
        # The stack trace shows it expects the 'ans' module here:
        destination_dir = os.path.join(task_root, "src/codec/entropy_coding/lib_wrappers/mans")
        
        # Check physically on disk for the compiled .so file
        compiled_extensions = glob.glob(os.path.join(destination_dir, "ans*.so"))
        
        if compiled_extensions:
            found_binary = os.path.basename(compiled_extensions[0])
            print(f"[{self.TASK_NAME}] Verification complete: Found compiled backend binary ({found_binary}).")
        else:
            print(f"\n[WARNING] {self.TASK_NAME} is missing its compiled ANS arithmetic coder extensions.")
            user_choice = input("Would you like to compile the arithmetic coder now for JPEG-AI? [y/N]: ").strip().lower()
            
            if user_choice in ['y', 'yes']:
                # Retrieve the explicit python environment path
                # Safely pull the env_path directly from the dictionary passed into the class
                target_env = self.params.get("env_path", "")
                target_env = os.path.abspath(os.path.expanduser(target_env))

                print(f"Compiling via Makefile: conda run --prefix {target_env} make build_test_libs")
                try:
                    # Run the specific make command from the root of the JPEG-AI repository
                    subprocess.run(
                        ["conda", "run", "--prefix", target_env, "make", "build_test_libs"], 
                        cwd=task_root, 
                        check=True
                    )
                    
                    print("[SUCCESS] JPEG-AI C++ libraries compiled successfully!")
                except subprocess.CalledProcessError as e:
                    print(f"[ERROR] Compilation failed with exit code {e.returncode}.")

            else:
                print("Skipping compilation. The upcoming training run will likely fail with a ModuleNotFoundError or ImportError.")