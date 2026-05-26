import subprocess
import os
import sys

def setup_conda_env(env_path: str, requirements_file: str, python_version: str = "3.10"):
    """
    Dynamically creates a Conda environment and installs requirements in an Ubuntu/Linux shell.
    """
    # Expand tildes (~) if the user references their Linux home directory
    env_path = os.path.abspath(os.path.expanduser(env_path))
    requirements_file = os.path.abspath(os.path.expanduser(requirements_file))

    if not os.path.exists(requirements_file):
        print(f"Error: Could not find '{requirements_file}'")
        sys.exit(1)

    print(f"Creating Conda environment at: {env_path} (Python {python_version})")
    
    # Explicitly include pip to keep it isolated to this environment prefix
    create_cmd = [
        "conda", "create",
        "--prefix", env_path,
        f"python={python_version}",
        "pip", 
        "-y"
    ]

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
    # Using 'python -m pip' instead of 'pip' command bypasses broken shebangs in env/bin/pip
    install_cmd = [
        "conda", "run",
        "--prefix", env_path,
        "python", "-m", "pip", "install", "-r", requirements_file
    ]
    
    try:
        subprocess.run(install_cmd, check=True)
        print("Dependencies verified/installed successfully!")
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