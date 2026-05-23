import os
import sys
import argparse
import importlib.util

# Dynamically import create-env.py because of the dash in filename
def load_create_env():
    script_path = os.path.join(os.path.dirname(__file__), "create-env.py")
    if not os.path.exists(script_path):
        print(f"Error: Could not find 'create-env.py' at {script_path}")
        sys.exit(1)
        
    spec = importlib.util.spec_from_file_location("create_env", script_path)
    create_env_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(create_env_mod)
    return create_env_mod

def main():
    parser = argparse.ArgumentParser(description="Quick-start environment setup for all integrated LIC models.")
    parser.add_argument("base_path", nargs="?", help="Base directory where environments will be created.")
    args = parser.parse_args()

    base_path = args.base_path
    if not base_path:
        base_path = input("Enter the base directory path for environments: ").strip()
    
    if not base_path:
        print("No path provided. Exiting.")
        sys.exit(1)

    base_path = os.path.abspath(os.path.expanduser(base_path))
    os.makedirs(base_path, exist_ok=True)

    create_env_mod = load_create_env()

    # Model configurations based on README.md and directory structure
    models = [
        {
            "name": "DCVC-RT",
            "python": "3.12",
            "req": "LIC-Models/DCVC-RT/requirements.txt"
        },
        {
            "name": "ELIC",
            "python": "3.10",
            "req": "LIC-Models/ELIC/requirements.txt"
        },
        {
            "name": "HPCM",
            "python": "3.8",
            "req": "LIC-Models/HPCM/requirements.txt"
        },
        {
            "name": "LIC-TCM",
            "python": "3.10",
            "req": "LIC-Models/LIC-TCM/requirements.txt"
        },
        {
            "name": "StableCodec",
            "python": "3.10",
            "req": "LIC-Models/StableCodec/requirements.txt"
        },
        {
            "name": "eval",
            "python": "3.10",
            "req": "evaluation-requirements.txt"
        }
    ]

    print(f"\nSetting up environments in: {base_path}\n")

    for model in models:
        env_name = f"{model['name']}-env"
        env_path = os.path.join(base_path, env_name)
        req_path = os.path.join(os.path.dirname(__file__), model['req'])
        py_ver = model['python']

        if not os.path.exists(req_path):
            print(f"[SKIP] Requirements file not found for {model['name']}: {req_path}")
            continue

        print("="*60)
        print(f"Installing {model['name']} Environment")
        print(f"Path: {env_path}")
        print(f"Python: {py_ver}")
        print("="*60)

        try:
            create_env_mod.setup_conda_env(env_path, req_path, py_ver)
        except Exception as e:
            print(f"\n[ERROR] Failed to setup environment for {model['name']}: {e}")
            choice = input("Continue with next model? (y/N): ").strip().lower()
            if choice != 'y':
                print("Aborting remaining setups.")
                break

    print("\nAll environment setup tasks completed.")

if __name__ == "__main__":
    main()
