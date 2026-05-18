import subprocess
import os
import sys

def setup_conda_env(env_path: str, requirements_file: str, python_version: str = "3.10"):
    """
    Dynamically creates a Conda environment and installs requirements.
    """
    # 1. Verify the requirements file exists before starting
    if not os.path.exists(requirements_file):
        print(f"Error: Could not find '{requirements_file}'")
        sys.exit(1)

    print(f"Creating Conda environment at: {env_path}")
    
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

        print(f"Installing dependencies from {requirements_file}...")
        
        # Command to install requirements.
        # Note: We use `conda run` to execute pip inside the newly created environment.
        # This handles packages that might only be on PyPI and not in Conda channels.
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

# --- Example Usage ---
if __name__ == "__main__":
    # Define your paths here (can be absolute or relative)
    TARGET_ENV_PATH = "./ELIC/training-testing-env"
    REQUIREMENTS_PATH = "./ELIC/requirements.txt"
    
    # You can quickly create a dummy requirements.txt for testing
    if not os.path.exists(REQUIREMENTS_PATH):
        with open(REQUIREMENTS_PATH, "w") as f:
            f.write("requests==2.31.0\nnumpy==1.26.0\n")
            
    setup_conda_env(TARGET_ENV_PATH, REQUIREMENTS_PATH)