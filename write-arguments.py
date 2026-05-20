import os
import sys
import json
import inspect
import argparse
import importlib.util

class ConfigGenerator:
    def __init__(self, train_interfaces_path, test_interfaces_path):
        self.registry = {}
        self.known_args = {}  # Tracks previously entered arguments to use as defaults
        
        if train_interfaces_path:
            self._load_interfaces_from_dir(train_interfaces_path, "Train")
        if test_interfaces_path:
            self._load_interfaces_from_dir(test_interfaces_path, "Test")

    def _load_interfaces_from_dir(self, directory, category):
        if not os.path.isdir(directory):
            print(f"[WARNING] {category} interface directory not found: {directory}")
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
                    except Exception as e:
                        print(f"  -> [WARNING] Failed to load {filename}: {e}")

    def _parse_type(self, value):
        """Converts user string input into proper int, float, or bool types for JSON."""
        val = value.strip()
        if not val:
            return None
            
        val_lower = val.lower()
        if val_lower == "true":
            return True
        if val_lower == "false":
            return False
            
        # Try int
        try:
            return int(val)
        except ValueError:
            pass
            
        # Try float
        try:
            return float(val)
        except ValueError:
            pass
            
        # Check for list syntax (e.g., "[256, 256]")
        if val.startswith("[") and val.endswith("]"):
            try:
                return json.loads(val)
            except json.JSONDecodeError:
                pass
                
        # Handle 'null' string as actual None
        if val_lower == "none" or val_lower == "null":
            return None
            
        return val # Default to string

    def _get_known_arg(self, arg, aliases):
        """Checks if a value for 'arg' or any of its aliases exists in memory."""
        # Direct match (e.g., 'dataset' is in global args)
        if arg in self.known_args:
            return self.known_args[arg]
            
        # Forward Lookup
        if arg in aliases:
            target = aliases[arg]
            if target in self.known_args:
                return self.known_args[target]
                
        # Reverse Lookup
        for alias_key, target in aliases.items():
            if target == arg:
                if alias_key in self.known_args:
                    return self.known_args[alias_key]                    
        return None

    def _prompt(self, message, default=None, required=False, is_path=False):
        """Helper to prompt the user for input, with optional path validation."""
        while True:
            prompt_str = f"{message} "
            if default is not None:
                default_type = type(default).__name__
                prompt_str += f"[Default: {default}] ({default_type}): "
            else:
                prompt_str += "(Required): " if required else "(Optional, press Enter to skip): "
                
            user_input = input(prompt_str)
            
            # Handle empty inputs
            if not user_input.strip():
                if default is not None:
                    val = default
                elif required:
                    print("  -> [ERROR] This argument is required. Please provide a value.")
                    continue
                else:
                    return None
            else:
                val = self._parse_type(user_input)

            # Handle path auto-resolution and validation safely
            if is_path and val is not None:
                expanded_path = os.path.abspath(os.path.expanduser(str(val)))
                
                if not os.path.exists(expanded_path):
                    print(f"  -> [WARNING] Path does not exist currently: '{expanded_path}'")
                    force = input("  -> Do you want to use this path anyway? (y/N): ").strip().lower()
                    if force != 'y':
                        continue
                        
                val = expanded_path  # Save the clean, absolute path
                
            return val

    def build_global_args(self):
        print("\n" + "="*50)
        print("Configuring Global Arguments")
        print("="*50)
        
        global_args = {}
        
        # Standard global arguments template with validation rules
        prompts = {
            # --- Hardware & Environment ---
            "cuda": {"default": True},
            "gpu_id": {"default": 0},
            "num_workers": {"default": 8},
            "seed": {"default": 42},
            
            # --- Datasets ---
            "train_dataset": {"default": None, "required": True, "is_path": True},
            "test_dataset": {"default": None, "required": True, "is_path": True},
            "train_split": {"default": ""},
            "test_split": {"default": ""},
            
            # --- Batching & Steps ---
            "epochs": {"default": 100},
            "batch_size": {"default": 16},
            "test_batch_size": {"default": 1},
            "patch_size": {"default": [256, 256]}, # Use actual list here!
            
            # --- Optimization ---
            "learning_rate": {"default": 1e-4},
            "aux_learning_rate": {"default": 1e-3},
            "clip_max_norm": {"default": 1.0},
            
            # --- Model Parameters ---
            "lambda": {"default": 0.01},
            "metrics": {"default": "mse"},
            
            # --- Logging & Checkpointing ---
            "output_directory": {"default": "default_run"},
            "save": {"default": True},
            "checkpoint": {"default": None, "is_path": True}
        }
        
        for key, opts in prompts.items():
            default_val = opts.get("default")
            is_req = opts.get("required", False)
            is_path = opts.get("is_path", False)
            
            val = self._prompt(f"Enter {key}", default=default_val, required=is_req, is_path=is_path)
            
            if val is not None:
                global_args[key] = val
                self.known_args[key] = val  # Save to memory
                
        return global_args

    def build_tasks(self):
        print("\n" + "="*50)
        print("Configuring Tasks")
        print("="*50)
        
        if not self.registry:
            print("No interfaces found! Please ensure your interface directories are correct.")
            return {}
            
        tasks = {}
        
        while True:
            # --- Status Dashboard ---
            print("\n" + "-"*45)
            print("AVAILABLE INTERFACES (Can be added):")
            for name in self.registry.keys():
                print(f"  [+] {name}")
                
            print("\nCURRENT JOB QUEUE (Already added):")
            if not tasks:
                print("  (Empty)")
            else:
                for t_key in tasks.keys():
                    print(f"  [*] {t_key}")
            print("-"*45)
            
            task_name = input("\nEnter the Task Name to add (or press Enter to finish): ").strip()
            
            if not task_name:
                break
                
            if task_name not in self.registry:
                print(f"  -> [ERROR] Unknown task type: '{task_name}'. Try again.")
                continue
                
            task_key = task_name
            existing_count = sum(1 for k in tasks.keys() if k.startswith(task_name))
            if existing_count > 0:
                task_key = f"{task_name}_{existing_count + 1}"
            
            print(f"\n--- Configuring Task: {task_key} ---")
            
            # 1. Base Task Information
            directory = self._prompt("Enter directory", default=task_name)
            env_path = self._prompt("Enter env_path", default=f"~/{directory}-env")
            
            task_info = {
                "env_path": env_path,
                "task_name": task_name,
                "directory": directory,
                "arguments": {}
            }
            
            interface_cls = self.registry[task_name]
            required_args = getattr(interface_cls, 'REQUIRED_ARGS', [])
            default_args = getattr(interface_cls, 'DEFAULT_VARS', {})
            aliases = getattr(interface_cls, 'ALIASES', {})  # Pull aliases for checking
            
            # 2. Required Arguments
            if required_args:
                print("\n[Required Arguments]")
                for arg in required_args:
                    prev_val = self._get_known_arg(arg, aliases)
                    
                    if prev_val is not None:
                        val = self._prompt(f"{arg}", default=prev_val)
                    else:
                        val = self._prompt(f"{arg}", required=True)
                        
                    task_info["arguments"][arg] = val
                    self.known_args[arg] = val  # Save to memory
            
            # 3. Optional / Default Arguments
            if default_args:
                print(f"\n[Default Arguments] This interface has {len(default_args)} default arguments.")
                override = input("Do you want to override any default arguments? (y/N): ").strip().lower()
                
                # BUG FIX: Pre-fill with defaults ONLY if not already answered in required_args
                for arg, default_val in default_args.items():
                    if arg not in required_args:
                        task_info["arguments"][arg] = default_val
                        
                if override == 'y':
                    for arg, default_val in default_args.items():
                        if arg in required_args:
                            continue 
                        
                        suggested_default = self._get_known_arg(arg, aliases)
                        if suggested_default is None:
                            suggested_default = default_val
                        
                        val = self._prompt(f"{arg}", default=suggested_default)
                        task_info["arguments"][arg] = val
                        self.known_args[arg] = val  # Save to memory
                else:
                    for arg, default_val in default_args.items():
                        if arg not in self.known_args:
                            self.known_args[arg] = default_val

            tasks[task_key] = task_info
            print(f"\nSuccessfully added {task_key} to job queue.")
            
        return tasks

    def generate(self, output_path="arguments.json"):
        global_args = self.build_global_args()
        tasks = self.build_tasks()
        
        final_config = {
            "global_arguments": global_args,
            "tasks": tasks
        }
        
        with open(output_path, 'w') as f:
            json.dump(final_config, f, indent=4)
            
        print("\n" + "="*50)
        print(f"Successfully generated {output_path}")
        print("="*50)


def parse_args():
    parser = argparse.ArgumentParser(description="Interactive Argument JSON Generator")
    parser.add_argument('--train_path', type=str, default="Interfaces/Training-Interfaces", help="Path to training interfaces")
    parser.add_argument('--test_path', type=str, default="Interfaces/Testing-Interfaces", help="Path to testing interfaces")
    parser.add_argument('--output', type=str, default="arguments.json", help="Path for the generated JSON file")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    generator = ConfigGenerator(
        train_interfaces_path=args.train_path,
        test_interfaces_path=args.test_path
    )
    
    generator.generate(output_path=args.output)