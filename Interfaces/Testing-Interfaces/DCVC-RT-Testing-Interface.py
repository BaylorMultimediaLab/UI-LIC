import os
import subprocess
from base_interface import BaseInterface

class DCVCRTImageTestInterface(BaseInterface):

    TASK_NAME = "DCVC-RT"
    
    USE_MODULE_EXECUTION = False
    EXECUTION_PATH = "Testing/test_image_encoding.py"
    REQUIRED_ARGS = ["model_path", "input", "save_dir"]
    ACTION_FLAGS = []

    DEFAULT_VARS = {
        "model_path": None,
        "input": None,
        "qp": 27,
        "device": "cuda",
        "save_dir": None
    }

    ALIASES = {
        "model": "model_path",
        "model_i": "model_path",
        "image": "input",
        "test_image": "input",
        "test_dataset": "input",
        "output_directory": "save_dir",
    }

    CLI_MAPPING = {
        "model_path": "--model_path",
        "input": "--input",
        "qp": "--qp",
        "device": "--device",
        "rec_path": "--rec_path",
        "bin_path": "--bin_path",
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

        # --- SAVE_DIR → REC/BIN SPLIT (IMPORTANT) ---
        if "save_dir" in self.params and self.params["save_dir"] is not None:
            base = self.params["save_dir"]

            self.params["rec_path"] = os.path.join(base, "reconstruction")
            self.params["bin_path"] = os.path.join(base, "bitstreams")

            # optional safety
            os.makedirs(self.params["rec_path"], exist_ok=True)
            os.makedirs(self.params["bin_path"], exist_ok=True)


    def _check_and_install_dependencies(self):
        """Checks the target ENV_PATH for required packages and prompts installation if missing."""
        python_exec = "python3"
        if hasattr(self, 'ENV_PATH') and self.ENV_PATH:
            python_exec = os.path.join(self.ENV_PATH, "bin", "python3")
            
        print(f"  -> Verifying dependencies in: {python_exec}")
        
        # 1. Check Standard Dependencies
        missing_standard = []
        for pkg, pip_name in [("torchvision", "torchvision"), ("PIL", "Pillow")]:
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
                    # ADDED --no-build-isolation HERE
                    subprocess.check_call([python_exec, "-m", "pip", "install", "--no-build-isolation", "."], cwd=cpp_path)
                else:
                    print(f"  -> [ERROR] Could not find path: {cpp_path}")
                
                print("  -> Compiling CUDA inference kernels...")
                if os.path.exists(cuda_path):
                    # ADDED --no-build-isolation HERE
                    subprocess.check_call([python_exec, "-m", "pip", "install", "--no-build-isolation", "."], cwd=cuda_path)
                else:
                    print(f"  -> [ERROR] Could not find path: {cuda_path}")
            else:
                print("  -> Proceeding anyway, but execution will likely fail.")

    def execute(self):
        """Overrides the base execute to ensure dependencies exist before running."""
        self._check_and_install_dependencies()
        super().execute()