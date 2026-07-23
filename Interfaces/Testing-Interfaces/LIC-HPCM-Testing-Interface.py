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

        work_dir = os.path.abspath(getattr(self, 'WORKING_DIR', 'LIC-Models/HPCM'))
        cpp_path = os.path.join(work_dir, "src", "entropy_models", "entropy_coders", "unbounded_rans")
        print("  -> [INFO] Compiling HPCM C++ arithmetic coding extension (_CXX)...")
        if os.path.exists(cpp_path):
            subprocess.check_call([python_exec, "-m", "pip", "install", "--no-build-isolation", "."], cwd=cpp_path)

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