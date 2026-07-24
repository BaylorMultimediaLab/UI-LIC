import os
import subprocess
from base_interface import BaseInterface

class DCVCRTImageTestInterface(BaseInterface):

    TASK_NAME = "DCVC-RT"
    ENV_PATH = "LIC-Models/DCVC-RT-env"
    WORKING_DIR = "LIC-Models/DCVC-RT"
    
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

        env = os.environ.copy()
        bin_dir = os.path.dirname(os.path.abspath(python_exec))
        venv_root = os.path.abspath(os.path.join(bin_dir, ".."))
        env["PATH"] = f"{bin_dir}{os.pathsep}{env.get('PATH', '')}"

        lib_dir = os.path.join(venv_root, "lib")
        if os.path.isdir(lib_dir):
            python_dirs = sorted([d for d in os.listdir(lib_dir) if d.startswith("python")], reverse=True)
            if python_dirs:
                site_packages = os.path.join(lib_dir, python_dirs[0], "site-packages")
                nvidia_dir = os.path.join(site_packages, "nvidia")
                if os.path.isdir(nvidia_dir):
                    cu_dirs = sorted([d for d in os.listdir(nvidia_dir) if d.startswith("cu") and d[2:].isdigit()], reverse=True)
                    if cu_dirs:
                        cuda_toolkit = os.path.join(nvidia_dir, cu_dirs[0])
                        nvcc_bin = os.path.join(cuda_toolkit, "bin")
                        if os.path.isdir(nvcc_bin):
                            env["PATH"] = f"{nvcc_bin}{os.pathsep}{env['PATH']}"
                        cuda_lib = os.path.join(cuda_toolkit, "lib")
                        if os.path.isdir(cuda_lib):
                            for f in os.listdir(cuda_lib):
                                if ".so." in f:
                                    base_name = f.split(".so.")[0] + ".so"
                                    so_path = os.path.join(cuda_lib, base_name)
                                    if not os.path.exists(so_path):
                                        try:
                                            os.symlink(f, so_path)
                                        except Exception:
                                            pass
                        if "CUDA_HOME" not in env:
                            env["CUDA_HOME"] = cuda_toolkit

        if "CUDA_HOME" not in env and os.path.exists("/usr/local/cuda"):
            env["CUDA_HOME"] = "/usr/local/cuda"
        if "CXX" not in env:
            env["CXX"] = "g++"
        if "TORCH_CUDA_ARCH_LIST" not in env:
            env["TORCH_CUDA_ARCH_LIST"] = "7.0;7.5;8.0;8.6;8.9;9.0"

        work_dir = os.path.abspath(getattr(self, 'WORKING_DIR', 'LIC-Models/DCVC-RT'))
        cpp_path = os.path.join(work_dir, "src", "cpp")
        cuda_path = os.path.join(work_dir, "src", "layers", "extensions", "inference")

        print("  -> Compiling C++ entropy coding extensions (this may take a minute)...")
        if os.path.exists(cpp_path):
            subprocess.check_call([python_exec, "-m", "pip", "install", "--no-build-isolation", "."], cwd=cpp_path, env=env)

        print("  -> Compiling CUDA inference kernels...")
        if os.path.exists(cuda_path):
            subprocess.check_call([python_exec, "-m", "pip", "install", "--no-build-isolation", "."], cwd=cuda_path, env=env)

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
            print(f"\n  -> [INFO] Auto-installing missing standard packages: {', '.join(missing_standard)}")
            subprocess.check_call([python_exec, "-m", "pip", "install", *missing_standard])

        # 2. Check Custom Extensions
        try:
            subprocess.check_call([python_exec, "-c", "import MLCodec_extensions_cpp"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            print("\n  -> [INFO] Missing custom extension 'MLCodec_extensions_cpp'. Compiling extensions automatically...")
            self.compile_extensions(python_exec=python_exec)

    def execute(self):
        """Overrides the base execute to ensure dependencies exist before running."""
        self._check_and_install_dependencies()
        super().execute()