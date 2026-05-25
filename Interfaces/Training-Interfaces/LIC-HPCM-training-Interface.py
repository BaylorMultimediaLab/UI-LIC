from base_interface import BaseInterface
import os
import sys

import subprocess

import importlib

class LIC_HCPM_TrainInterface(BaseInterface):

    TASK_NAME = "HPCM"
    
    USE_MODULE_EXECUTION = False

    # Update this if your execution file is named differently
    EXECUTION_PATH = "train.py"
    
    # Enforce these to avoid accidental default overwrites. 
    # model_name, train_dataset, and test_dataset are strictly necessary for the script to execute.
    REQUIRED_ARGS = ["model_name", "train_dataset", "test_dataset"]

    ## Model name can be [HPCM_Base/HPCM_Large]
    
    # Tells BaseInterface NOT to append "True" or "False" to these boolean flags.
    # If they exist in the JSON as 'true', they will be passed as empty switches (e.g., --save)
    ACTION_FLAGS = [
        "save"
    ]

    # Derived from the explicit argparse usage inside the provided LIC-HCPM training script
    DEFAULT_VARS = {
        "model_name": "HPCM_base",
        "model_class": "hypers",
        "train_dataset": None,
        "test_dataset": None,
        "epochs": 3001,
        "learning_rate": 1e-4,
        "num_workers": 8,
        "lmbda": 0.013,
        "batch_size": 32,
        "test_batch_size": 1,
        "aux_learning_rate": 1e-3,
        "patch_size": [256, 256],
        "cuda": True,
        "save": True,
        "save_path": "output/",
        "log_dir": "output/",
        "seed": None,
        "clip_max_norm": 1.0,
        "checkpoint": None
    }

    # Map unified standard names from your JSON to LIC-HCPM's specific variable names
    ALIASES = {
        "lambda_rate": "lmbda",
        "lr": "learning_rate",
        "epochs": "epochs",
        "workers": "num_workers"
    }

    # Map the internal configuration standard names to the argparse CLI flags
    CLI_MAPPING = {
        "model_name": "--model_name",
        "model_class": "--model_class",
        "train_dataset": "--train_dataset",
        "test_dataset": "--test_dataset",
        "epochs": "--epochs",
        "learning_rate": "--learning-rate",
        "num_workers": "--num-workers",
        "lmbda": "--lambda",  # Note: The dest is 'lmbda', but the CLI expects '--lambda'
        "batch_size": "--batch-size",
        "test_batch_size": "--test-batch-size",
        "aux_learning_rate": "--aux-learning-rate",
        "patch_size": "--patch-size",
        "cuda": "--cuda",
        "save": "--save",
        "save_path": "--save_path",
        "log_dir": "--log_dir",
        "seed": "--seed",
        "clip_max_norm": "--clip_max_norm",
        "checkpoint": "--checkpoint"
    }

    def __init__(self, job_args=None, global_args=None):
        # Call the parent BaseInterface init to load and merge all arguments
        super().__init__(job_args, global_args)


        for key, val in self.params.items():
            if isinstance(val, bool):
                self.params[key] = 1 if val else 0
        
        
        # Translate unified integer/scalar patch sizes (e.g., 256) to a list/tuple of two ints
        # since the target argparse expects nargs=2 (e.g., --patch-size 256 256)
        if "patch_size" in self.params:
            val = self.params["patch_size"]
            if isinstance(val, int):
                # Expand single int to [int, int]
                self.params["patch_size"] = [val, val]
            elif isinstance(val, str):
                try:
                    # E.g., "256" -> [256, 256]
                    v = int(val.strip())
                    self.params["patch_size"] = [v, v]
                except ValueError:
                    # Handles if the string was passed as "256 256"
                    parts = val.split()
                    if len(parts) >= 2:
                        self.params["patch_size"] = [int(parts[0]), int(parts[1])]
                    elif len(parts) == 1:
                        self.params["patch_size"] = [int(parts[0]), int(parts[0])]


        # --- RECOMMENDED HPCM C++ COMPILATION ---
        
        task_root = "LIC-HPCM"
        destination_dir = os.path.join(task_root, "src/entropy_models")
        
        # Check physically on disk for the compiled .so file
        # This prevents Python 3.13 (dispatcher) from failing to "import" a Python 3.8 binary
        import glob
        compiled_extensions = glob.glob(os.path.join(destination_dir, "_CXX*.so"))
        
        if compiled_extensions:
            found_binary = os.path.basename(compiled_extensions[0])
            print(f"[{self.TASK_NAME}] Verification complete: Found compiled backend binary ({found_binary}).")
        else:
            print(f"\n[WARNING] {self.TASK_NAME} is missing its compiled arithmetic coder extensions.")
            user_choice = input("Would you like to compile the arithmetic coder now using HPCM's build configuration? [y/N]: ").strip().lower()
            
            if user_choice in ['y', 'yes']:
                target_script_dir = os.path.join(task_root, "src/entropy_models/entropy_coders/unbounded_rans")
                
                if not os.path.exists(target_script_dir):
                    print(f"[ERROR] Could not find recommended directory path: {target_script_dir}")
                    return

                # Retrieve the explicit python environment path
                target_env = self.params.get("env_path", "")
                target_env = os.path.abspath(os.path.expanduser(target_env))
                env_python = os.path.join(target_env, "bin/python3")

                if not os.path.exists(env_python):
                    print(f"[ERROR] Could not locate Python binary at: {env_python}")
                    return

                print(f"Compiling directly via target binary: {env_python} setup.py build_ext --inplace")
                try:
                    if os.path.exists(os.path.join(target_script_dir, "setup.py")):
                        subprocess.run(
                            [env_python, "setup.py", "build_ext", "--inplace"], 
                            cwd=target_script_dir, 
                            check=True
                        )
                    else:
                        subprocess.run(
                            ["conda", "run", "--prefix", target_env, "sh", "setup.sh"], 
                            cwd=target_script_dir, 
                            check=True
                        )
                    
                    # Relocate the files
                    import shutil
                    so_files = glob.glob(os.path.join(target_script_dir, "*.so"))
                    for so_file in so_files:
                        shutil.copy(so_file, destination_dir)
                        print(f"[INTERFACE] Relocated {os.path.basename(so_file)} -> {destination_dir}")

                    print("[SUCCESS] Arithmetic coder compiled and relocated successfully!")
                except subprocess.CalledProcessError as e:
                    print(f"[ERROR] Compilation failed with exit code {e.returncode}.")
            else:
                print("Skipping compilation. The upcoming training run will likely fail with a ModuleNotFoundError.")