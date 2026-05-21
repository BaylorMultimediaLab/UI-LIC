import os
import subprocess
from base_interface import BaseInterface

class DCVCRTTrainInterface(BaseInterface):

    TASK_NAME = "DCVC-RT"
    
    USE_MODULE_EXECUTION = False
    EXECUTION_PATH = "Training/train_dcvc_rt_intra.py"
    REQUIRED_ARGS = ["train_root"]
    ACTION_FLAGS = []

    DEFAULT_VARS = {
        "train_root": None,
        "val_root": None,
        "out_dir": "Training/runs/intra",
        "patch_size": 256,
        "batch_size": 8,
        "num_workers": 2,
        "lr": 1e-4,
        "lambda_rd": 0.01,
        "epochs": 50,          
        "qp": -1,
        "device": "cuda",
        "seed": 0,
        "log_every": 50,
        "save_every": 1000,
        "val_every": 1000,
        "val_batches": 25
    }

    ALIASES = {
        "train_dataset": "train_root",
        "test_dataset": "val_root",
        "save_path": "out_dir",
        "learning_rate": "lr",
        "lambda": "lambda_rd",
        "lmbda": "lambda_rd",
        "bs": "batch_size",
        "e": "epochs"        
    }

    CLI_MAPPING = {
        "train_root": "--train_root",
        "val_root": "--val_root",
        "out_dir": "--out_dir",
        "patch_size": "--patch_size",
        "batch_size": "--batch_size",
        "num_workers": "--num_workers",
        "lr": "--lr",
        "lambda_rd": "--lambda_rd",
        "epochs": "--epochs",  
        "qp": "--qp",
        "device": "--device",
        "seed": "--seed",
        "log_every": "--log_every",
        "save_every": "--save_every",
        "val_every": "--val_every",
        "val_batches": "--val_batches"
    }
    
    def __init__(self, job_args=None, global_args=None):
        super().__init__(job_args, global_args)

        # --- UNIFIED TRANSLATION LOGIC ---
        if "cuda" in self.params:
            if self.params["cuda"] is True:
                self.params["device"] = "cuda"
            elif self.params["cuda"] is False:
                self.params["device"] = "cpu"
            del self.params["cuda"]
            
        if "patch_size" in self.params and isinstance(self.params["patch_size"], (list, tuple)):
            self.params["patch_size"] = self.params["patch_size"][0]

    def _check_and_install_dependencies(self):
        """Checks the target ENV_PATH for required packages and prompts installation if missing."""
        python_exec = "python3"
        if hasattr(self, 'ENV_PATH') and self.ENV_PATH:
            python_exec = os.path.join(self.ENV_PATH, "bin", "python3")
            
        print(f"  -> Verifying dependencies in: {python_exec}")
        
        # 1. Check Standard Dependencies
        missing_standard = []
        for pkg, pip_name in [("torchvision", "torchvision"), ("PIL", "Pillow"), ("pybind11", "pybind11")]:
            try:
                subprocess.check_call([python_exec, "-c", f"import {pkg}"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except subprocess.CalledProcessError:
                missing_standard.append(pip_name)
                
        if missing_standard:
            print(f"\n  -> [WARNING] Missing standard packages: {', '.join(missing_standard)}")
            choice = input(f"  -> Would you like to install them now? (y/N): ").strip().lower()
            if choice == 'y':
                subprocess.check_call([python_exec, "-m", "pip", "install", *missing_standard])
            else:
                print("  -> Proceeding anyway, but execution will likely fail.")

        # 2. Check Custom Extensions
        try:
            subprocess.check_call([python_exec, "-c", "import MLCodec_extensions_cpp"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            print("\n  -> [WARNING] Missing custom extension: 'MLCodec_extensions_cpp'")
            print("  -> DCVC-RT requires custom C++ and CUDA extensions to be compiled locally.")
            choice = input("  -> Would you like to compile and install them automatically now? (y/N): ").strip().lower()
            
            if choice == 'y':
                work_dir = getattr(self, 'WORKING_DIR', '.')
                cpp_path = os.path.join(work_dir, "src", "cpp")
                cuda_path = os.path.join(work_dir, "src", "layers", "extensions", "inference")
                
                print("  -> Compiling C++ entropy coding extensions (this may take a minute)...")
                if os.path.exists(cpp_path):
                    subprocess.check_call([python_exec, "-m", "pip", "install", "--no-build-isolation", "."], cwd=cpp_path)
                else:
                    print(f"  -> [ERROR] Could not find path: {cpp_path}")
                
                print("  -> Compiling CUDA inference kernels...")
                if os.path.exists(cuda_path):
                    subprocess.check_call([python_exec, "-m", "pip", "install", "--no-build-isolation", "."], cwd=cuda_path)
                else:
                    print(f"  -> [ERROR] Could not find path: {cuda_path}")
            else:
                print("  -> Proceeding anyway, but execution will likely fail.")

    def execute(self):
        """Overrides the base execute to ensure dependencies exist before running."""
        self._check_and_install_dependencies()
        super().execute()