"""
Unified Interface For Learned Image Compression (UI-LIC) - Environment Setup Utility

This script provides automated creation and updating of isolated Conda environments for individual LIC models.
It creates Python environments at specified target directories (`--prefix`) and installs model-specific dependencies
from `requirements.txt` files to avoid library version conflicts between different research codebases.
"""

import subprocess
import os
import sys

def build_model_extensions(env_path: str, requirements_file: str):
    """
    Automatically compile model-specific C++ / CUDA extensions after pip install.

    This function scans all Testing-Interface files for classes that define a
    compile_extensions() method, matches them to the model being installed (by
    comparing the requirements file's parent folder name against the interface's
    ENV_PATH, WORKING_DIR, or filename), and runs the compilation step.

    This is called at the end of setup_conda_env() and update_pip_requirements()
    so that extensions like DCVC-RT's rans or HPCM's unbounded_rans are built
    automatically instead of requiring manual compilation.
    """
    env_path = os.path.abspath(os.path.expanduser(env_path))

    # Auto-repair existing/legacy Conda environments missing the C++ compiler toolchain on Linux
    if sys.platform.startswith("linux"):
        gxx_bin = os.path.join(env_path, "bin", "x86_64-conda-linux-gnu-g++")
        if not os.path.exists(gxx_bin):
            print("Installing C++ compiler toolchain (gxx_linux-64) to ensure pybind11 compatibility...")
            try:
                subprocess.run([
                    "conda", "install",
                    "--prefix", env_path,
                    "-c", "conda-forge",
                    "gxx_linux-64", "sysroot_linux-64",
                    "-y"
                ], check=True)
            except subprocess.CalledProcessError as e:
                print(f"[WARNING] Could not auto-install gxx_linux-64: {e}")

    # Set CUDA_HOME, CXX, and TORCH_CUDA_ARCH_LIST environment variables if missing
    if "CUDA_HOME" not in os.environ and os.path.exists("/usr/local/cuda"):
        os.environ["CUDA_HOME"] = "/usr/local/cuda"
    if "CXX" not in os.environ:
        os.environ["CXX"] = "g++"
    if "TORCH_CUDA_ARCH_LIST" not in os.environ:
        os.environ["TORCH_CUDA_ARCH_LIST"] = "7.0;7.5;8.0;8.6;8.9;9.0"

    # Resolve the venv's Python executable (Linux bin/ or Windows Scripts/)
    python_exec = os.path.join(env_path, "bin", "python3")
    if not os.path.exists(python_exec):
        python_exec = os.path.join(env_path, "Scripts", "python.exe")

    # Auto-create unversioned .so symlinks in site-packages/nvidia (e.g. libcudart.so -> libcudart.so.13)
    bin_dir = os.path.dirname(os.path.abspath(python_exec))
    venv_root = os.path.abspath(os.path.join(bin_dir, ".."))
    lib_dir = os.path.join(venv_root, "lib")
    if os.path.isdir(lib_dir):
        python_dirs = sorted([d for d in os.listdir(lib_dir) if d.startswith("python")], reverse=True)
        if python_dirs:
            site_packages = os.path.join(lib_dir, python_dirs[0], "site-packages")
            nvidia_dir = os.path.join(site_packages, "nvidia")
            if os.path.isdir(nvidia_dir):
                for root, dirs, files in os.walk(nvidia_dir):
                    for f in files:
                        if ".so." in f:
                            base_name = f.split(".so.")[0] + ".so"
                            so_path = os.path.join(root, base_name)
                            if not os.path.exists(so_path):
                                try:
                                    os.symlink(f, so_path)
                                except Exception:
                                    pass

    # Derive the model folder name from the requirements file path.
    # e.g. "LIC-Models/DCVC-RT/requirements.txt" -> "dcvc-rt"
    req_dir = os.path.dirname(os.path.abspath(requirements_file))
    folder_name = os.path.basename(req_dir).lower()

    interfaces_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "Interfaces", "Testing-Interfaces"))
    if not os.path.exists(interfaces_dir):
        return

    import importlib.util

    # Add interface directories to sys.path so base_interface imports resolve
    sys.path.insert(0, os.path.dirname(interfaces_dir))
    sys.path.insert(0, interfaces_dir)

    for fname in os.listdir(interfaces_dir):
        if fname.endswith(".py") and not fname.startswith("__"):
            mod_name = fname[:-3]
            try:
                spec = importlib.util.spec_from_file_location(mod_name, os.path.join(interfaces_dir, fname))
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                for attr_name in dir(mod):
                    attr = getattr(mod, attr_name)
                    if isinstance(attr, type) and hasattr(attr, "compile_extensions"):
                        # Match this interface class to the model being installed
                        # by checking if the folder name appears in ENV_PATH,
                        # WORKING_DIR, or the interface filename.
                        env_p = getattr(attr, "ENV_PATH", "") or ""
                        work_p = getattr(attr, "WORKING_DIR", "") or ""
                        if folder_name and (folder_name in env_p.lower() or folder_name in work_p.lower() or folder_name in fname.lower()):
                            try:
                                # Use __new__ to create the instance without
                                # calling __init__, which may raise due to
                                # missing required args (e.g. HPCM needs
                                # checkpoint and dataset). compile_extensions()
                                # only needs class-level attrs like WORKING_DIR.
                                inst = attr.__new__(attr)
                                inst.compile_extensions(python_exec=python_exec)
                            except Exception as e:
                                print(f"  -> [WARNING] compile_extensions failed for {attr_name}: {e}")
            except Exception as e:
                print(f"  -> [WARNING] Failed to load interface module {fname}: {e}")

def setup_conda_env(env_path: str, requirements_file: str, python_version: str = "3.10"):
    """
    Dynamically creates a Conda environment at a specified path prefix and installs requirements.
    
    Using --prefix ensures environments are stored cleanly within specified project paths rather
    than global conda environment directories.
    """
    # Expand tildes (~) if the user references their Linux home directory
    env_path = os.path.abspath(os.path.expanduser(env_path))
    requirements_file = os.path.abspath(os.path.expanduser(requirements_file))

    if not os.path.exists(requirements_file):
        print(f"Error: Could not find '{requirements_file}'")
        sys.exit(1)

    print(f"Creating Conda environment at: {env_path} (Python {python_version})")
    
    # Explicitly include pip and C++ compiler toolchain (on Linux) to ensure pybind11/C++ extensions compile cleanly and isolate packages
    create_cmd = [
        "conda", "create",
        "--prefix", env_path,
        "-c", "conda-forge",
        "-y",
        f"python={python_version}",
        "pip"
    ]
    if sys.platform.startswith("linux"):
        create_cmd.extend(["gxx_linux-64", "sysroot_linux-64"])

    try:
        print("Running environment creation...")
        subprocess.run(create_cmd, check=True) # Clean execution, no shell=True needed on Linux
        print("Environment created successfully.")

        print(f"Installing remaining dependencies from {requirements_file}...")
        install_cmd = [
            "conda", "run",
            "--prefix", env_path,
            "pip", "install", "-r", requirements_file
        ]
        
        subprocess.run(install_cmd, check=True)
        print("Dependencies installed successfully!")
        
        # Automate compilation of model-specific C++ / CUDA extensions
        build_model_extensions(env_path, requirements_file)

    except subprocess.CalledProcessError as e:
        print(f"\n[!] A sub-process failed with exit code {e.returncode}.")
        print(f"Command run: {' '.join(e.cmd)}")
        sys.exit(1)
    except FileNotFoundError:
        print("\n[!] Conda is not recognized as a command.")
        print("Make sure Conda is initialized in your Ubuntu shell (run 'conda init bash' and restart the shell if needed).")
        sys.exit(1)

def update_pip_requirements(env_path: str, requirements_file: str):
    """
    Installs/Updates requirements in an existing Conda environment.
    """
    env_path = os.path.abspath(os.path.expanduser(env_path))
    requirements_file = os.path.abspath(os.path.expanduser(requirements_file))

    if not os.path.exists(env_path):
        print(f"Error: Environment not found at {env_path}")
        return

    if not os.path.exists(requirements_file):
        print(f"Error: Could not find '{requirements_file}'")
        return

    print(f"Ensuring dependencies from {requirements_file} in {env_path}...")
    
    # Auto-repair existing/legacy Conda environments missing the C++ compiler toolchain on Linux
    if sys.platform.startswith("linux"):
        gxx_bin = os.path.join(env_path, "bin", "x86_64-conda-linux-gnu-g++")
        if not os.path.exists(gxx_bin):
            print("Installing C++ compiler toolchain (gxx_linux-64) to ensure pybind11 compatibility...")
            try:
                subprocess.run([
                    "conda", "install",
                    "--prefix", env_path,
                    "-c", "conda-forge",
                    "gxx_linux-64", "sysroot_linux-64",
                    "-y"
                ], check=True)
            except subprocess.CalledProcessError as e:
                print(f"[WARNING] Could not auto-install gxx_linux-64: {e}")

    # Using 'python -m pip' instead of 'pip' command bypasses broken shebangs in env/bin/pip
    install_cmd = [
        "conda", "run",
        "--prefix", env_path,
        "python", "-m", "pip", "install", "-r", requirements_file
    ]
    
    try:
        subprocess.run(install_cmd, check=True)
        print("Dependencies verified/installed successfully!")
        build_model_extensions(env_path, requirements_file)
    except subprocess.CalledProcessError as e:
        print(f"[WARNING] Failed to update dependencies: {e}")

if __name__ == "__main__":    
    TARGET_ENV_PATH = input("Enter target environment path: ").strip()
    if not TARGET_ENV_PATH:
        print("\n[CANCELLED] No environment path provided. Exiting script.")
        sys.exit(0)
        
    # Get Requirements Path
    REQUIREMENTS_PATH = input("Enter requirements file path: ").strip()
    if not REQUIREMENTS_PATH:
        print("\n[CANCELLED] No requirements path provided. Exiting script.")
        sys.exit(0)
        
    # Get Python Version
    PYTHON_VERSION = input("Enter Python version [Default: 3.10]: ").strip()
    if not PYTHON_VERSION:
        PYTHON_VERSION = "3.10"
        
    print("-" * 50)
    print(f"Using Environment Path: {TARGET_ENV_PATH}")
    print(f"Using Requirements Path: {REQUIREMENTS_PATH}")
    print(f"Using Python Version:    {PYTHON_VERSION}")
    print("-" * 50)
                
    setup_conda_env(TARGET_ENV_PATH, REQUIREMENTS_PATH, PYTHON_VERSION)