import subprocess
import os
import sys

def setup_conda_env(env_path: str, requirements_file: str, python_version: str = "3.10"):
    """
    Dynamically creates a Conda environment and installs requirements.
    """
    # Verify the requirements file exists before starting
    if not os.path.exists(requirements_file):
        print(f"Error: Could not find '{requirements_file}'")
        sys.exit(1)

    print(f"Creating Conda environment at: {env_path} (Python {python_version})")
    
    # Command to create the environment
    create_cmd = [
        "conda", "create",
        "--prefix", env_path,
        f"python={python_version}",
        "-y" # Automatically say yes to prompts
    ]

    try:
        # Run the creation command
        subprocess.run(create_cmd, check=True)
        print("Environment created successfully.")

        print(f"Installing remaining dependencies from {requirements_file}...")
        install_cmd = [
            "conda", "run",
            "--prefix", env_path,
            "pip", "install", "-r", requirements_file
        ]
        
        # Run the installation command
        subprocess.run(install_cmd, check=True)
        print("Dependencies installed successfully!")

    except subprocess.CalledProcessError as e:
        print(f"\n[!] A sub-process failed with exit code {e.returncode}.")
        print(f"Command run: {' '.join(e.cmd)}")
        sys.exit(1)
    except FileNotFoundError:
        print("\n[!] Conda is not recognized as a command. Ensure it is installed and in your system PATH.")
        sys.exit(1)

if __name__ == "__main__":
    print("--- Environment Setup Configuration ---")
    print("(Press Enter on an empty line at an env or requirements prompt to cancel and exit)")
    print("-" * 40)
    
    # 1. Get Environment Path
    TARGET_ENV_PATH = input("Enter target environment path: ").strip()
    if not TARGET_ENV_PATH:
        print("\n[CANCELLED] No environment path provided. Exiting script.")
        sys.exit(0)
        
    # 2. Get Requirements Path
    REQUIREMENTS_PATH = input("Enter requirements file path: ").strip()
    if not REQUIREMENTS_PATH:
        print("\n[CANCELLED] No requirements path provided. Exiting script.")
        sys.exit(0)
        
    # 3. Get Python Version (Optional, defaults to 3.10)
    PYTHON_VERSION = input("Enter Python version [Default: 3.10]: ").strip()
    if not PYTHON_VERSION:
        PYTHON_VERSION = "3.10"
        
    print("-" * 40)
    print(f"Using Environment Path: {TARGET_ENV_PATH}")
    print(f"Using Requirements Path: {REQUIREMENTS_PATH}")
    print(f"Using Python Version:    {PYTHON_VERSION}")
    print("-" * 40)
                
    setup_conda_env(TARGET_ENV_PATH, REQUIREMENTS_PATH, PYTHON_VERSION)