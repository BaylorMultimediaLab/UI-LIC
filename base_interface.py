"""
Unified Interface For Learned Image Compression (UI-LIC) - Base Interface Module

This module provides the BaseInterface abstract base class that standardizes job execution,
argument mapping, and command-line translation across all integrated learned image compression models.
It enables seamless translation of unified UI-LIC configuration parameters into model-specific CLI flags
and manages isolated Conda environment execution.
"""

import subprocess
import os
import platform

class BaseInterface:
    """
    Abstract Base Class for model training and testing interfaces.
    
    Handles parameter normalization (alias resolution), validation of required arguments,
    dynamic CLI command construction (including boolean action switches and lists), and
    subprocess execution within targeted Python environment contexts.
    """
    REQUIRED_ARGS = []
    DEFAULT_VARS = {}
    ALIASES = {}
    CLI_MAPPING = {}
    ACTION_FLAGS = []  # Defines which booleans act as CLI switches (flag present without trailing value)
    EXECUTION_PATH = ""
    
    WORKING_DIR = None
    USE_MODULE_EXECUTION = False

    def __init__(self, job_args=None, global_args=None):
        job_args = job_args or {}
        global_args = global_args or {}
        
        self.params = self.DEFAULT_VARS.copy()
        
        # Normalize incoming argument keys using model-specific ALIASES dictionary
        clean_job_args = {self.ALIASES.get(k, k): v for k, v in job_args.items()}
        clean_global_args = {self.ALIASES.get(k, k): v for k, v in global_args.items()}

        # Merge global defaults first, then override with task-specific job parameters
        self.params.update(clean_global_args)    
        self.params.update(clean_job_args)
        
        # Parse string representations of booleans ("true"/"false") into PyTorch/Python bools
        for key, value in self.params.items():
            if isinstance(value, str):
                cleaned_val = value.strip().lower()
                if cleaned_val == "true":
                    self.params[key] = True
                elif cleaned_val == "false":
                    self.params[key] = False

    def validate(self):
        missing = [req for req in self.REQUIRED_ARGS if self.params.get(req) is None]
        if missing:
            return False, missing
        return True, []

    def build_command(self):
        if hasattr(self, 'ENV_PATH') and self.ENV_PATH:
            if self.WORKING_DIR:
                abs_env = os.path.abspath(os.path.join(self.WORKING_DIR, self.ENV_PATH))
            else:
                abs_env = os.path.abspath(self.ENV_PATH)
            
            if os.path.isdir(os.path.join(abs_env, "Scripts")):
                python_executable = os.path.join(abs_env, "Scripts", "python.exe")
            else:
                python_executable = os.path.join(abs_env, "bin", "python3")
                
            command = [python_executable]
        else:
            command = ["python"] if platform.system() == "Windows" else ["python3"]        
        
        if self.USE_MODULE_EXECUTION:
            command.extend(["-m", self.EXECUTION_PATH])
        else:
            command.append(self.EXECUTION_PATH)

        # The single, unified argument builder loop
        for standard_key, cli_flag in self.CLI_MAPPING.items():
            if standard_key in self.params:
                val = self.params[standard_key]
                if val is None or val == "":
                    continue
                
                # Check if this flag is registered as a switch 
                if standard_key in getattr(self, 'ACTION_FLAGS', []):
                    if val is True: 
                        command.append(cli_flag)
                        
                # Handle lists/tuples
                elif isinstance(val, (list, tuple)):
                    command.append(cli_flag)
                    command.extend([str(v) for v in val])
                    
                # Handle standard key-values
                else:
                    command.append(cli_flag)
                    command.append(str(val))
                    
        return command
    
    def execute(self):
        command = self.build_command()
        print(f"Executing:\n{' '.join(command)}\n")
        
        run_dir = os.path.abspath(self.WORKING_DIR) if self.WORKING_DIR else os.getcwd()
        
        run_env = os.environ.copy()

        # --- CUDA / JIT compilation environment setup --------------------------
        if command and os.path.exists(command[0]):
            bin_dir = os.path.dirname(os.path.abspath(command[0]))
            run_env["PATH"] = f"{bin_dir}{os.pathsep}{run_env.get('PATH', '')}"

            venv_root = os.path.abspath(os.path.join(bin_dir, ".."))
            lib_dir = os.path.join(venv_root, "lib")
            site_packages = None
            if os.path.isdir(lib_dir):
                python_dirs = sorted(
                    [d for d in os.listdir(lib_dir) if d.startswith("python")],
                    reverse=True
                )
                if python_dirs:
                    site_packages = os.path.join(lib_dir, python_dirs[0], "site-packages")

            if site_packages:
                nvidia_dir = os.path.join(site_packages, "nvidia")
                cuda_toolkit = None
                if os.path.isdir(nvidia_dir):
                    cu_dirs = sorted(
                        [d for d in os.listdir(nvidia_dir)
                         if d.startswith("cu") and d[2:].isdigit()],
                        reverse=True
                    )
                    if cu_dirs:
                        cuda_toolkit = os.path.join(nvidia_dir, cu_dirs[0])

                if cuda_toolkit:
                    nvcc_bin = os.path.join(cuda_toolkit, "bin")
                    if os.path.isdir(nvcc_bin):
                        run_env["PATH"] = f"{nvcc_bin}{os.pathsep}{run_env['PATH']}"

                    cuda_lib = os.path.join(cuda_toolkit, "lib")
                    if os.path.isdir(cuda_lib):
                        try:
                            for f in os.listdir(cuda_lib):
                                if ".so." in f:
                                    base_name = f.split(".so.")[0] + ".so"
                                    so_path = os.path.join(cuda_lib, base_name)
                                    if not os.path.exists(so_path):
                                        os.symlink(f, so_path)
                        except Exception:
                            pass

                    if "CUDA_HOME" not in run_env:
                        run_env["CUDA_HOME"] = cuda_toolkit

            if "CUDA_HOME" not in run_env and os.path.exists("/usr/local/cuda"):
                run_env["CUDA_HOME"] = "/usr/local/cuda"

        if "CXX" not in run_env:
            run_env["CXX"] = "g++"

        if "TORCH_CUDA_ARCH_LIST" not in run_env:
            run_env["TORCH_CUDA_ARCH_LIST"] = "7.0;7.5;8.0;8.6;8.9;9.0"

        existing_pythonpath = run_env.get("PYTHONPATH", "")
        
        if existing_pythonpath:
            run_env["PYTHONPATH"] = f"{run_dir}{os.pathsep}{existing_pythonpath}"
        else:
            run_env["PYTHONPATH"] = run_dir
        
        try:
            subprocess.run(command, check=True, text=True, cwd=run_dir, env=run_env)
            print("[SUCCESS] Script completed.")
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Execution failed with code {e.returncode}")
            raise e