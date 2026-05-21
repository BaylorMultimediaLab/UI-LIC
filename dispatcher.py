import os
import sys
import json
import time
import argparse
import inspect
import importlib.util
import shutil

#
# Reads training and testing interfaces, requires variables to be passed for selected
# interfaces.
# 
#

class Dispatcher:
    def __init__(self, arg_json_path, train_interfaces_path=None, test_interfaces_path=None):
        self.arg_json_path = arg_json_path
        self.registry = {}

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

                # Dynamically load the module
                spec = importlib.util.spec_from_file_location(module_name, filepath)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    try:
                        spec.loader.exec_module(module)
                        
                        # Inspect the module for classes with 'TASK_NAME'
                        for name, obj in inspect.getmembers(module, inspect.isclass):
                            if hasattr(obj, 'TASK_NAME') and getattr(obj, 'TASK_NAME') is not None:
                                self.registry[obj.TASK_NAME] = obj
                                print(f"  -> Registered '{obj.TASK_NAME}' from {filename}")
                                
                    except Exception as e:
                        print(f"  -> [WARNING] Failed to load {filename}: {e}")


    def _verify_crop_sizes(self, data_dir, crop_size):
        """Scans data_dir for images smaller than crop_size and offers to remove them."""
        if not data_dir or not os.path.isdir(data_dir) or not crop_size:
            return

        # Normalize crop_size whether it is an int (320) or a list ([256, 256])
        if isinstance(crop_size, int):
            min_w = min_h = crop_size
        elif isinstance(crop_size, list) and len(crop_size) >= 2:
            min_w, min_h = crop_size[0], crop_size[1]
        else:
            return  # Unknown crop_size format

        try:
            from PIL import Image
        except ImportError:
            print("  -> [WARNING] 'Pillow' is not installed. Skipping crop size verification.")
            return

        invalid_files = []
        valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')
        
        print(f"  -> Verifying images in {data_dir} are at least {min_w}x{min_h}...")
        
        # Scan directory
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
                        pass # Ignore corrupted files here; let the dataloader handle them
        
        if invalid_files:
            print(f"\n  -> [WARNING] Found {len(invalid_files)} images smaller than the crop size.")
            print("     These will likely crash the PyTorch dataloader.")
            choice = input("     Would you like to move these out of the training folder? (y/N): ")
            
            if choice.lower() == 'y':
                custom_path = input(f"     Enter a destination path (or press Enter to default to '../skipped_undersized_{min_w}x{min_h}'): ").strip()
                
                if custom_path:
                    # Resolve '~/' and make absolute
                    skipped_dir = os.path.abspath(os.path.expanduser(custom_path))
                else:
                    # Default safely outside the train directory
                    skipped_dir = os.path.abspath(os.path.join(data_dir, "..", f"skipped_undersized_{min_w}x{min_h}"))
                
                os.makedirs(skipped_dir, exist_ok=True)
                
                for fpath in invalid_files:
                    filename = os.path.basename(fpath)
                    dest = os.path.join(skipped_dir, filename)
                    
                    # Prevent overwriting if two files have the same name in different subfolders
                    if os.path.exists(dest):
                        dest = os.path.join(skipped_dir, f"{int(time.time())}_{filename}")
                        
                    shutil.move(fpath, dest)
                print(f"  -> Successfully moved {len(invalid_files)} images to: {skipped_dir}\n")
            else:
                print("  -> Leaving files untouched. Execution may fail if the model cannot pad them.\n")

    def _normalize_booleans(self, args_dict):
        """Helper to recursively scan and turn 'True'/'False' strings into Python booleans."""
        normalized = {}
        for k, v in args_dict.items():
            if isinstance(v, str):
                cleaned = v.strip().lower()
                if cleaned == "true":
                    normalized[k] = True
                elif cleaned == "false":
                    normalized[k] = False
                else:
                    normalized[k] = v
            elif isinstance(v, dict):
                normalized[k] = self._normalize_booleans(v)
            else:
                normalized[k] = v
        return normalized
    
    def _expand_paths(self, args_dict):
        """Helper to recursively scan and expand any '~' home directory paths in string values."""
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
        # Load arguments from JSON
        try:
            with open(self.arg_json_path, 'r') as file:
                config = json.load(file)
        # TODO: make interactive interface for when arguments file is not provided
        # -> No args.json found, would you like to create one?
        # -> No args.json found, would you like to start a session by inputting values??
        except FileNotFoundError:
                    print(f"[FATAL] Argument file '{self.arg_json_path}' not found. Exiting.")
                    return
        
        global_args = config.get("global_arguments", {})
        tasks = config.get("tasks", {})
        
        if not tasks:
            print("[WARNING] No jobs found in the 'jobs' dictionary. Nothing to execute.")
            return
        
        clean_global_args = self._normalize_booleans(global_args)
        clean_global_args = self._expand_paths(clean_global_args)  # <--- Clean path pass


        for step, (task_key, task_info) in enumerate(tasks.items(), start=1):
            # print("\n" + "="*50)
            # print(f"Step {step}/{len(job_queue)}: Starting [{task_name}]")
            # print("="*50)
            
            target_task_name = task_info.get("task_name", task_key)
            custom_dir = task_info.get("directory")
            
            env_path = task_info.get("env_path")
            job_args = task_info.get("arguments", {})

            # Check if the requested string matches a loaded Interface's task_name
            if target_task_name not in self.registry:
                print(f"  -> [SKIPPED] Unknown task type: '{target_task_name}'.")
                print(f"     (Make sure an interface file with TASK_NAME = '{target_task_name}' exists).")
                continue
            
            clean_job_args = self._normalize_booleans(job_args)
            clean_job_args = self._expand_paths(clean_job_args)  # <--- Clean path pass

            # We pass the entire global pool. The interface's ALIASES will safely extract only what it needs.
            InterfaceClass = self.registry[target_task_name]
            interface_instance = InterfaceClass(job_args=clean_job_args, global_args=clean_global_args)            
            
            # Override the Interface's default directory if one was provided in the JSON
            if custom_dir:
                clean_custom_dir = os.path.abspath(os.path.expanduser(custom_dir))
                interface_instance.WORKING_DIR = clean_custom_dir                 
            if env_path:
                clean_env_path = os.path.abspath(os.path.expanduser(env_path))
                interface_instance.ENV_PATH = clean_env_path
                
            target_data_dir = clean_job_args.get("data_dir") or clean_global_args.get("train_dataset")
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


def parse_args():
    parser = argparse.ArgumentParser(description="Unified Dispatcher for Training and Testing")
    
    parser.add_argument('--train', action='store_true', help="Enable training phase")
    parser.add_argument('--test', action='store_true', help="Enable testing phase")

    parser.add_argument('--train_path', type=str, default="Interfaces/Training-Interfaces", help="Path to training interfaces")
    parser.add_argument('--test_path', type=str, default="Interfaces/Testing-Interfaces", help="Path to testing interfaces")
    parser.add_argument('--args_json', type=str, default="arguments.json", help="Path to the arguments JSON file")

    return parser.parse_args()


if __name__ == "__main__":
    

    begin_time = time.time()

    args = parse_args()

    if args.train and not os.path.isdir(args.train_path):
        print(f"FATAL: You requested --train, but the directory '{args.train_path}' does not exist.")
        sys.exit(1)
        
    if args.test and not os.path.isdir(args.test_path):
        print(f"FATAL: You requested --test, but the directory '{args.test_path}' does not exist.")
        sys.exit(1)


    dispatcher = Dispatcher(
        arg_json_path=args.args_json,
        train_interfaces_path=args.train_path, #if args.train else None,
        test_interfaces_path=args.test_path, # if args.test else None
    )
    dispatcher.run()

    print(f"\nTotal Execution Time: {time.time() - begin_time:.2f} seconds")
