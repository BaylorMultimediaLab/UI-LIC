"""
Unified Interface For Learned Image Compression (UI-LIC) - Execution Dispatcher Engine

This module (`dispatcher.py`) serves as the central execution engine of the UI-LIC framework. It reads task queues
from JSON configuration files (`arguments.json`), performs pre-execution safety checks (dataset path verification and
minimum image patch size validation), switches into model-specific Conda environments, executes training or testing
jobs via unified interface wrappers, and automatically invokes `evaluation.py` to calculate perceptual and quantitative metrics.
"""

import os
import sys
import json
import time
import argparse
import subprocess
import inspect
import importlib.util
import shutil

class Dispatcher:
    """
    Job Dispatcher and Execution Manager.
    
    Parses configured job queues, verifies dataset/checkpoint existence interactively before execution,
    prevents PyTorch dataloader crashes by validating image dimensions against model patch requirements,
    and manages hands-free evaluation pipeline invocation upon completion.
    """
    def __init__(self, arg_json_path, run_train=False, run_test=False, train_interfaces_path=None, test_interfaces_path=None):
        self.arg_json_path = arg_json_path
        self.registry = {}
        
        self.run_train = run_train
        self.run_test = run_test

        if not os.path.exists(self.arg_json_path):
            raise FileNotFoundError(f"Error: Argument file '{self.arg_json_path}' not found.")

        if train_interfaces_path:
            self._load_interfaces_from_dir(train_interfaces_path, "Train")
            
        if test_interfaces_path:
            self._load_interfaces_from_dir(test_interfaces_path, "Test")

    def _load_interfaces_from_dir(self, directory, category):
        if not os.path.isdir(directory):
            print(f"[ERROR] {category} interface directory not found: {directory}")
            return

        for filename in os.listdir(directory):
            if filename.endswith(".py") and not filename.startswith("__"):
                filepath = os.path.join(directory, filename)
                module_name = filename[:-3]

                spec = importlib.util.spec_from_file_location(module_name, filepath)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    try:
                        spec.loader.exec_module(module)
                        for name, obj in inspect.getmembers(module, inspect.isclass):
                            if hasattr(obj, 'TASK_NAME') and getattr(obj, 'TASK_NAME') is not None:
                                self.registry[obj.TASK_NAME] = obj
                                print(f"  -> Registered '{obj.TASK_NAME}' from {filename}")
                    except Exception as e:
                        print(f"  -> [WARNING] Failed to load {filename}: {e}")

    def _validate_paths_interactive(self, args_dict, depth=0):
        if depth >= 3:
            return args_dict
            
        # Keys that are almost always file system paths
        path_keys = {
            "dataset", "data", "input_dir", "input_dirs", 
            "checkpoint", "checkpoints", "save_dir", "output_dir", 
            "data_dir", "log_dir", "sd_path", "elic_path", "test_dataset", "train_dataset"
        }
        
        for k, v in args_dict.items():
            if k in path_keys and isinstance(v, str):
                path = os.path.expanduser(v)
                
                # If it's a save/output directory, we can usually just create it
                if "save" in k or "output" in k:
                    if not os.path.exists(path):
                        print(f" -> [INFO] Path '{k}' ('{path}') does not exist. Creating it.")
                        os.makedirs(path, exist_ok=True)
                
                # If it's an input/dataset/checkpoint, it MUST exist
                elif not os.path.exists(path):
                    print(f"\n -> [WARNING] Path for '{k}' ('{path}') NOT FOUND.")
                    if k == 'log_dir':
                        new_path = 'skip'
                    else:
                        new_path = input(f"    Please enter the correct path for '{k}' (or 'skip'): ").strip()
                    
                    if new_path.lower() != 'skip' and new_path != "":
                        args_dict[k] = new_path
                        # Recursive check on the new path
                        self._validate_paths_interactive({k: new_path}, depth + 1)
        return args_dict

    def _verify_crop_sizes(self, data_dir, crop_size):
        if not data_dir or not os.path.isdir(data_dir) or not crop_size:
            return

        if isinstance(crop_size, int):
            min_w = min_h = crop_size
        elif isinstance(crop_size, list) and len(crop_size) >= 2:
            min_w, min_h = crop_size[0], crop_size[1]
        else:
            return  

        try:
            from PIL import Image
        except ImportError:
            print("  -> [WARNING] 'Pillow' is not installed. Skipping crop size verification.")
            return

        invalid_files = []
        valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')
        
        print(f"  -> Verifying images in {data_dir} are at least {min_w}x{min_h}...")
        
        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.lower().endswith(valid_extensions):
                    filepath = os.path.join(root, file)
                    try:
                        with Image.open(filepath) as img:
                            w, h = img.size
                            if w < min_w or h < min_h:
                                invalid_files.append(filepath)
                    except Exception:
                        pass 
        
        if invalid_files:
            print(f"\n  -> [WARNING] Found {len(invalid_files)} images smaller than the crop size.")
            print("     These will likely crash the PyTorch dataloader.")
            choice = input("     Would you like to move these out of the dataset folder? (y/N): ")
            
            if choice.lower() == 'y':
                custom_path = input(f"     Enter a destination path (or press Enter to default to '../skipped_undersized_{min_w}x{min_h}'): ").strip()
                if custom_path:
                    skipped_dir = os.path.abspath(os.path.expanduser(custom_path))
                else:
                    skipped_dir = os.path.abspath(os.path.join(data_dir, "..", f"skipped_undersized_{min_w}x{min_h}"))
                
                os.makedirs(skipped_dir, exist_ok=True)
                
                for fpath in invalid_files:
                    filename = os.path.basename(fpath)
                    dest = os.path.join(skipped_dir, filename)
                    if os.path.exists(dest):
                        dest = os.path.join(skipped_dir, f"{int(time.time())}_{filename}")
                    shutil.move(fpath, dest)
                print(f"  -> Successfully moved {len(invalid_files)} images to: {skipped_dir}\n")
            else:
                print("  -> Leaving files untouched. Execution may fail if the model cannot pad them.\n")

    def _normalize_booleans(self, args_dict):
        normalized = {}
        for k, v in args_dict.items():
            if isinstance(v, str):
                cleaned = v.strip().lower()
                if cleaned == "true": normalized[k] = True
                elif cleaned == "false": normalized[k] = False
                else: normalized[k] = v
            elif isinstance(v, dict):
                normalized[k] = self._normalize_booleans(v)
            else:
                normalized[k] = v
        return normalized
    
    def _expand_paths(self, args_dict):
        expanded = {}
        for k, v in args_dict.items():
            if isinstance(v, str) and v.startswith("~/"):
                expanded[k] = os.path.expanduser(v)
            elif isinstance(v, dict):
                expanded[k] = self._expand_paths(v)
            else:
                expanded[k] = v
        return expanded
    
    def run(self):
        try:
            with open(self.arg_json_path, 'r') as file:
                config = json.load(file)
        except FileNotFoundError:
            print(f"[FATAL] Argument file '{self.arg_json_path}' not found. Exiting.")
            return

        # 1. Determine which phases to run based on initialization flags
        phases_to_run = []
        if self.run_train: phases_to_run.append("training")
        if self.run_test: phases_to_run.append("testing")

        if not phases_to_run:
            print("[WARNING] No execution phases were selected. Exiting.")
            return

        # 2. Iterate through only the selected phases
        for phase in phases_to_run:
            print("\n" + "="*60)
            print(f"=== Executing Phase: {phase.upper()} ===")
            print("="*60)
            
            # Extract the nested block (e.g., config["training"])
            phase_data = config.get(phase, {})
            global_args = phase_data.get("global_arguments", {})
            tasks = phase_data.get("tasks", {})

            if not tasks:
                print(f"[WARNING] No tasks found for '{phase}' in the JSON. Skipping.")
                continue

            clean_global_args = self._normalize_booleans(global_args)
            clean_global_args = self._expand_paths(clean_global_args)

            for step, (task_key, task_info) in enumerate(tasks.items(), start=1):
                print(f"\n--- [Phase: {phase.upper()}] Step {step}/{len(tasks)}: Starting {task_key} ---")
                
                target_task_name = task_info.get("task_name", task_key)
                custom_dir = task_info.get("directory")
                env_path = task_info.get("env_path")
                job_args = task_info.get("arguments", {})

                if target_task_name not in self.registry:
                    print(f"  -> [SKIPPED] Unknown task type: '{target_task_name}'.")
                    print(f"     (Make sure an interface file with TASK_NAME = '{target_task_name}' is loaded).")
                    continue
                
                clean_job_args = self._normalize_booleans(job_args)
                clean_job_args = self._expand_paths(clean_job_args)

                # Validate global args first, then job specific args
                global_args = self._validate_paths_interactive(clean_global_args)
                clean_job_args = self._validate_paths_interactive(clean_job_args)

                InterfaceClass = self.registry[target_task_name]
                interface_instance = InterfaceClass(job_args=clean_job_args, global_args=clean_global_args)            
                
                if custom_dir:
                    clean_custom_dir = os.path.abspath(os.path.expanduser(custom_dir))
                    interface_instance.WORKING_DIR = clean_custom_dir                
                if env_path:
                    clean_env_path = os.path.abspath(os.path.expanduser(env_path))
                    interface_instance.ENV_PATH = clean_env_path
                
                # 3. Dynamic dataset fallback depending on the current phase
                dataset_key = "train_dataset" if phase == "training" else "test_dataset"
                target_data_dir = clean_job_args.get("data_dir") or clean_global_args.get(dataset_key)
                target_crop_size = clean_job_args.get("crop_size") or clean_global_args.get("patch_size")
                
                if target_data_dir and target_crop_size:
                    self._verify_crop_sizes(target_data_dir, target_crop_size)
                    
                is_valid, missing_args = interface_instance.validate()
                
                if not is_valid:
                    print(f"  -> [FAILED] Interface '{target_task_name}' is missing required arguments: {missing_args}")
                    print(f"  -> Skipping to next job...")
                    continue

                print("  -> [VERIFIED] All required arguments present. Executing...\n")
                try:
                    interface_instance.execute()
                except Exception as e:
                    print(f"\n  -> [ERROR] Execution of '{target_task_name}' crashed: {e}")
                    print("  -> Continuing to next job in the queue...")

        # --- AUTOMATED EVALUATION STEP ---
        if phase == "testing":
            eval_data = phase_data.get("evaluation", {})
            if eval_data and "tasks" in eval_data:
                # 1. Resolve the evaluation environment
                eval_env = eval_data.get("env_path", None)
                if eval_env and eval_env != "n/a":
                    abs_eval_env = os.path.expanduser(eval_env)
                    if os.path.isdir(os.path.join(abs_eval_env, "Scripts")):
                        python_exec = os.path.join(abs_eval_env, "Scripts", "python.exe")
                    else:
                        python_exec = os.path.join(abs_eval_env, "bin", "python3")
                else:
                    python_exec = sys.executable

                print("\n" + "="*60)
                print(f"=== Executing Post-Test Evaluation (Env: {eval_env}) ===")
                print("="*60)
                
                for job in eval_data["tasks"]:
                    # Safely get required fields with defaults to prevent NoneType errors
                    task_name = job.get("task_name")
                    save_dir = job.get("save_dir")
                    input_dir = job.get("input_dir")
                    use_vmaf = job.get("use_vmaf", eval_data.get("use_vmaf", False))

                    if not all([task_name, save_dir, input_dir]):
                        print(f" -> [SKIP] Evaluation task missing required fields: {job}")
                        continue

                    print(f"\n--- Evaluating Task: {task_name} ---")
                    
                    cmd = [
                        python_exec, "evaluation.py",
                        "--task_name", str(task_name),
                        "--save_dir", str(save_dir),
                        "--input_dir", str(input_dir)
                    ]
                    
                    if use_vmaf:
                        cmd.append("--use_vmaf")
                    
                    try:
                        subprocess.run(cmd, check=True)
                    except subprocess.CalledProcessError as e:
                        print(f" -> [ERROR] Evaluation script failed for {task_name}: {e}")


def parse_args():
    parser = argparse.ArgumentParser(description="Unified Dispatcher for Training and Testing")
    parser.add_argument('--train', action='store_true', help="Enable training phase execution")
    parser.add_argument('--test', action='store_true', help="Enable testing phase execution")
    parser.add_argument('--train_path', type=str, default="Interfaces/Training-Interfaces", help="Path to training interfaces")
    parser.add_argument('--test_path', type=str, default="Interfaces/Testing-Interfaces", help="Path to testing interfaces")
    parser.add_argument('--args_json', type=str, default="arguments.json", help="Path to the arguments JSON file")
    return parser.parse_args()


if __name__ == "__main__":
    begin_time = time.time()
    args = parse_args()

    # --- JSON INTELLIGENCE ---
    # If no flags are provided, peek into the JSON to see what is available
    if not args.train and not args.test:
        has_train_tasks = False
        has_test_tasks = False
        
        try:
            with open(args.args_json, 'r') as file:
                config = json.load(file)
                
                # Check if 'training' exists and actually contains tasks
                if "training" in config and config["training"].get("tasks"):
                    has_train_tasks = True
                    
                # Check if 'testing' exists and actually contains tasks
                if "testing" in config and config["testing"].get("tasks"):
                    has_test_tasks = True
                    
        except FileNotFoundError:
            print(f"[FATAL] Argument file '{args.args_json}' not found. Please generate it first.")
            sys.exit(1)
        except json.JSONDecodeError:
            print(f"[FATAL] Argument file '{args.args_json}' is not valid JSON. Please check formatting.")
            sys.exit(1)
            
        # Decision Logic based on JSON contents
        if not has_train_tasks and not has_test_tasks:
            print(f"[FATAL] No training or testing tasks found in '{args.args_json}'. Exiting.")
            sys.exit(1)
            
        elif has_train_tasks and has_test_tasks:
            print("\nBoth Training and Evaluation tasks found. Please select an option:")
            print("  1) Execute Train and Evaluation")
            print("  2) Execute Training only")
            print("  3) Execute Evaluation only")
            
            while True:
                choice = input("Enter choice (1/2/3): ").strip()
                if choice == '1':
                    args.train = True
                    args.test = True
                    break
                elif choice == '2':
                    args.train = True
                    break
                elif choice == '3':
                    args.test = True
                    break
                else:
                    print("Invalid choice, please enter 1, 2, or 3.")
                    
        elif has_train_tasks and not has_test_tasks:
            choice = input("\nFound ONLY Training tasks. Do you want to execute them? (y/N): ").strip().lower()
            if choice == 'y':
                args.train = True
            else:
                print("Execution cancelled.")
                sys.exit(0)
                
        elif has_test_tasks and not has_train_tasks:
            choice = input("\nFound ONLY Evaluation tasks. Do you want to execute them? (y/N): ").strip().lower()
            if choice == 'y':
                args.test = True
            else:
                print("Execution cancelled.")
                sys.exit(0)
                
        print() # Empty line for spacing

    # --- DIRECTORY VERIFICATION ---
    if args.train and not os.path.isdir(args.train_path):
        print(f"FATAL: You requested training, but the directory '{args.train_path}' does not exist.")
        sys.exit(1)
        
    if args.test and not os.path.isdir(args.test_path):
        print(f"FATAL: You requested evaluation, but the directory '{args.test_path}' does not exist.")
        sys.exit(1)

    # --- DISPATCHER INITIALIZATION ---
    dispatcher = Dispatcher(
        arg_json_path=args.args_json,
        run_train=args.train,
        run_test=args.test,
        train_interfaces_path=args.train_path if args.train else None,
        test_interfaces_path=args.test_path if args.test else None
    )
    
    dispatcher.run()

    print(f"\nTotal Execution Time: {time.time() - begin_time:.2f} seconds")