import os
import subprocess
from base_interface import BaseInterface


class HPCMTestInterface(BaseInterface):

    TASK_NAME = "HPCM"
    ENV_PATH = "LIC-Models/HPCM-env"
    WORKING_DIR = "LIC-Models/HPCM"

    USE_MODULE_EXECUTION = False
    EXECUTION_PATH = "test.py"  # your HPCM eval script

    REQUIRED_ARGS = ["model_name", "checkpoint", "dataset", "save_dir"]

    ACTION_FLAGS = []

    # model name can be HPCM_Base or HPCM_Large
    DEFAULT_VARS = {
        "model_name": "HPCM_Base",
        "checkpoint": None,
        "dataset": None,
        "num": 60,

        # root output dir (contains reconstructed/ + latents/)
        "save_dir": "./outputs"
    }

    # -------------------------
    # Aliases (unified naming style)
    # -------------------------
    ALIASES = {
        "m": "model_name",
        "model": "model_name",

        "ckpt": "checkpoint",

        "data": "dataset",
        "test_dataset": "dataset",
        "input": "dataset",

        "out": "save_dir",
        "output_dir": "save_dir",
    }

    # -------------------------
    # CLI mapping (matches HPCM script)
    # -------------------------
    CLI_MAPPING = {
        "model_name": "--model_name",
        "checkpoint": "--checkpoint",
        "dataset": "--dataset",
        "num": "-num",
        "save_dir": "--save_dir",
    }

    def __init__(self, job_args=None, global_args=None):
        super().__init__(job_args, global_args)

        if not self.params.get("dataset"):
            raise ValueError("dataset is required")

        if not self.params.get("checkpoint"):
            raise ValueError("checkpoint is required")

        # Always align model_name with checkpoint filename (HPCM_Large vs HPCM_Base)
        ckpt = self.params["checkpoint"]
        ckpt_str = str(ckpt[0] if isinstance(ckpt, list) and ckpt else ckpt).lower()
        if "large" in ckpt_str:
            self.params["model_name"] = "HPCM_Large"
        elif "base" in ckpt_str:
            self.params["model_name"] = "HPCM_Base"

        if self.params.get("save_dir"):
            os.makedirs(self.params["save_dir"], exist_ok=True)

    def compile_extensions(self, python_exec=None):
        """Compiles HPCM C++ arithmetic coding extension (_CXX / HT)."""
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

        work_dir = os.path.abspath(getattr(self, 'WORKING_DIR', 'LIC-Models/HPCM'))
        cpp_path = os.path.join(work_dir, "src", "entropy_models", "entropy_coders", "unbounded_rans")
        print("  -> [INFO] Compiling HPCM C++ arithmetic coding extension (_CXX)...")
        if os.path.exists(cpp_path):
            subprocess.check_call([python_exec, "-m", "pip", "install", "--no-build-isolation", "."], cwd=cpp_path, env=env)

    def _check_and_install_dependencies(self):
        """Ensures the HPCM C++ extension (_CXX / HT) is compiled in HPCM-env."""
        python_exec = "python3"
        if hasattr(self, 'ENV_PATH') and self.ENV_PATH:
            abs_env = os.path.abspath(self.ENV_PATH)
            if os.path.isdir(os.path.join(abs_env, "bin")):
                python_exec = os.path.join(abs_env, "bin", "python3")

        try:
            subprocess.check_call([python_exec, "-c", "import _CXX"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            self.compile_extensions(python_exec=python_exec)

    def execute(self):
        """
        Ensure HPCM eval compatibility before execution.
        """
        self._check_and_install_dependencies()

        # -------------------------
        # checkpoint validation
        # -------------------------
        ckpt = self.params.get("checkpoint")

        if ckpt is None:
            raise ValueError("checkpoint is required")

        # allow single or list, but normalize to string (HPCM uses single ckpt loop externally)
        if isinstance(ckpt, list):
            if len(ckpt) == 1:
                self.params["checkpoint"] = ckpt[0]
            else:
                # optional: allow multi-ckpt runs by passing first one or raise
                self.params["checkpoint"] = ckpt

        # -------------------------
        # ensure save_dir exists (root output)
        # -------------------------
        os.makedirs(self.params["save_dir"], exist_ok=True)

        super().execute()