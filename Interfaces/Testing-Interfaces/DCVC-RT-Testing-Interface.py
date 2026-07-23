"""
Unified Interface For Learned Image Compression (UI-LIC) - DCVC-RT Testing Interface

This testing interface bridges UI-LIC parameters to DCVC-RT's image evaluation script (`Testing/test_image_encoding.py`).
It translates standard UI-LIC CLI arguments (such as `save_dir`, `cuda`, `qp`, `model_path`) into model-specific flags
and structures reconstructed images and bitstreams into uniform subdirectories for automated evaluation.
"""

import os
import subprocess
from base_interface import BaseInterface

class DCVCRTImageTestInterface(BaseInterface):
    """
    Testing Interface for DCVC-RT (Deep Contextual Video Compression - Real Time) evaluation.
    """

    TASK_NAME = "DCVC-RT"
    
    USE_MODULE_EXECUTION = False
    EXECUTION_PATH = "Testing/test_image_encoding.py"
    REQUIRED_ARGS = ["model_path", "input", "save_dir"]
    ACTION_FLAGS = []

    DEFAULT_VARS = {
        "model_path": None,
        "input": None,
        "qp": 47,
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

            os.makedirs(self.params["rec_path"], exist_ok=True)
            os.makedirs(self.params["bin_path"], exist_ok=True)

    def build_command(self):
        if "save_dir" in self.params and self.params["save_dir"] is not None:
            base = self.params["save_dir"]
            self.params["rec_path"] = os.path.join(base, "reconstruction")
            self.params["bin_path"] = os.path.join(base, "bitstreams")
            os.makedirs(self.params["rec_path"], exist_ok=True)
            os.makedirs(self.params["bin_path"], exist_ok=True)
        return super().build_command()


    def compile_extensions(self, python_exec=None):
        """Compiles DCVC-RT C++ and CUDA extensions."""
        if not python_exec:
            python_exec = "python3"
            if hasattr(self, 'ENV_PATH') and self.ENV_PATH:
                abs_env = os.path.abspath(self.ENV_PATH)
                if os.path.isdir(os.path.join(abs_env, "bin")):
                    python_exec = os.path.join(abs_env, "bin", "python3")

        work_dir = os.path.abspath(getattr(self, 'WORKING_DIR', 'LIC-Models/DCVC-RT'))
        cpp_path = os.path.join(work_dir, "src", "cpp")
        cuda_path = os.path.join(work_dir, "src", "layers", "extensions", "inference")

        print("  -> Compiling C++ entropy coding extensions (this may take a minute)...")
        if os.path.exists(cpp_path):
            subprocess.check_call([python_exec, "-m", "pip", "install", "--no-build-isolation", "."], cwd=cpp_path)

        print("  -> Compiling CUDA inference kernels...")
        if os.path.exists(cuda_path):
            subprocess.check_call([python_exec, "-m", "pip", "install", "--no-build-isolation", "."], cwd=cuda_path)

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
                self.compile_extensions(python_exec=python_exec)
            else:
                print("  -> Proceeding anyway, but execution will likely fail.")

    def execute(self):
        """Overrides the base execute to ensure dependencies exist before running."""
        self._check_and_install_dependencies()
        super().execute()