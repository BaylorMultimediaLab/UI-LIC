import os
import sys
import json
import inspect
import argparse
import importlib.util

class ConfigGenerator:
    def __init__(self, train_interfaces_path, test_interfaces_path, train_mode=False, test_mode=False):
        self.train_registry = {}
        self.test_registry = {}
        self.known_args = {}  # Tracks previously entered arguments to use as defaults
        
        self.train_mode = train_mode
        self.test_mode = test_mode
        
        if train_interfaces_path:
            self._load_interfaces_from_dir(train_interfaces_path, self.train_registry, "Train")
        if test_interfaces_path:
            self._load_interfaces_from_dir(test_interfaces_path, self.test_registry, "Test")

    def _load_interfaces_from_dir(self, directory, registry, category):
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
                                registry[obj.TASK_NAME] = obj
                    except Exception as e:
                        print(f"  -> [WARNING] Failed to load {filename}: {e}")

    def _parse_type(self, value):
        val = value.strip()
        if not val: return None
        val_lower = val.lower()
        if val_lower == "true": return True
        if val_lower == "false": return False
        
        try: return int(val)
        except ValueError: pass
        
        try: return float(val)
        except ValueError: pass
        
        if val.startswith("[") and val.endswith("]"):
            try: return json.loads(val)
            except json.JSONDecodeError: pass
                
        if val_lower == "none" or val_lower == "null":
            return None
            
        return val

    def _get_known_arg(self, arg, aliases):
        if arg in self.known_args:
            return self.known_args[arg]
        if arg in aliases:
            target = aliases[arg]
            if target in self.known_args:
                return self.known_args[target]
        for alias_key, target in aliases.items():
            if target == arg:
                if alias_key in self.known_args:
                    return self.known_args[alias_key]                    
        return None

    def _matches_global(self, arg, val, global_args, aliases):
        if arg in global_args and global_args[arg] == val: return True
        if arg in aliases:
            target = aliases[arg]
            if target in global_args and global_args[target] == val: return True
        for alias_key, target in aliases.items():
            if target == arg:
                if alias_key in global_args and global_args[alias_key] == val: return True
        return False

    def _prompt(self, message, default=None, required=False, is_path=False):
        while True:
            prompt_str = f"{message} "
            if default is not None:
                default_type = type(default).__name__
                prompt_str += f"[Default: {default}] ({default_type}): "
            else:
                prompt_str += "(Required): " if required else "(Optional, press Enter to skip): "
                
            user_input = input(prompt_str)
            
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

            if is_path and val is not None:
                expanded_path = os.path.abspath(os.path.expanduser(str(val)))
                if not os.path.exists(expanded_path):
                    print(f"  -> [WARNING] Path does not exist currently: '{expanded_path}'")
                    force = input("  -> Do you want to use this path anyway? (y/N): ").strip().lower()
                    if force != 'y': continue
                val = expanded_path  
                
            return val

    def build_global_args(self, phase):
        print("\n" + "="*50)
        print(f"Configuring Global Arguments [{phase.upper()}]")
        print("="*50)
        
        global_args = {}
        prompts = {}

        # --- Base Environment (Always Asked) ---
        prompts["cuda"] = {"default": True}
        prompts["gpu_id"] = {"default": 0}

        if phase == "training":
            prompts["num_workers"] = {"default": 8}
            prompts["seed"] = {"default": 42}
            prompts["train_dataset"] = {"default": None, "required": True, "is_path": True}
            prompts["train_split"] = {"default": "", "message": "Enter train_split (e.g., 'train', or leave empty if dataset root contains images)"}
            prompts["epochs"] = {"default": 100}
            prompts["batch_size"] = {"default": 16}
            prompts["patch_size"] = {"default": [256, 256]}
            prompts["learning_rate"] = {"default": 1e-4}
            prompts["aux_learning_rate"] = {"default": 1e-3}
            prompts["clip_max_norm"] = {"default": 1.0}
            prompts["lambda"] = {"default": 0.01}
            prompts["save"] = {"default": True}
            prompts["checkpoint"] = {"default": None, "is_path": True, "message": "Enter checkpoint file path"}
            prompts["metrics"] = {"default": "mse"}
            prompts["output_directory"] = {"default": "checkpoints"}
            
        elif phase == "testing":
            prompts["num_workers"] = {"default": 1}
            prompts["test_dataset"] = {"default": None, "required": True, "is_path": True}
            prompts["qp"] = {"default": 27}
            prompts["save_decoded_frame"] = {"default": False}
            prompts["metrics"] = {"default": "mse"}
            prompts["save_dir"] = {"default": "checkpoints", "is_path": True}
            prompts["output_images_directory"] = {"default": "eval_images", "is_path": True}
            prompts["output_metrics_directory"] = {"default": "eval_metrics", "is_path": True}
        
        for key, opts in prompts.items():
            default_val = opts.get("default")
            
            # If the user answered this in the previous phase (e.g. cuda), pre-fill it!
            if key in self.known_args:
                default_val = self.known_args[key]
                
            is_req = opts.get("required", False)
            is_path = opts.get("is_path", False)
            msg = opts.get("message", f"Enter {key}")
            
            val = self._prompt(msg, default=default_val, required=is_req, is_path=is_path)
            
            if val is not None:
                global_args[key] = val
                self.known_args[key] = val  
                
        return global_args

    def build_tasks(self, phase, global_args, omit_globals=False):
        print("\n" + "="*50)
        print(f"Configuring Tasks [{phase.upper()}]")
        print("="*50)
        
        eval_records = []

        registry = self.train_registry if phase == "training" else self.test_registry
        
        if not registry:
            print(f"No {phase} interfaces found! Skipping...")
            return {}, []
            
        tasks = {}
        
        while True:
            print("\n" + "-"*45)
            print(f"AVAILABLE {phase.upper()} INTERFACES:")
            for name in registry.keys():
                print(f"  [+] {name}")
                
            print("\nCURRENT JOB QUEUE (Already added):")
            if not tasks:
                print("  (Empty)")
            else:
                for t_key in tasks.keys():
                    print(f"  [*] {t_key}")
            print("-" * 45)
            
            task_name = input("\nEnter the Task Name to add (from the list above, or press Enter to finish phase): ").strip()
            
            if not task_name:
                break
                
            if task_name not in registry:
                print(f"  -> [ERROR] Unknown task type: '{task_name}'. Try again.")
                continue
                
            task_key = task_name
            existing_count = sum(1 for k in tasks.keys() if k.startswith(task_name))
            if existing_count > 0:
                task_key = f"{task_name}_{existing_count + 1}"
            
            print(f"\n--- Configuring Task: {task_key} ---")
            
            directory = self._prompt("Enter directory", default=f"LIC-Models/{task_name}")
            env_path = self._prompt("Enter env_path", default=f"LIC-Models/{task_name}-env")
            
            task_info = {
                "env_path": env_path,
                "task_name": task_name,
                "directory": directory,
                "arguments": {}
            }
            
            

            interface_cls = registry[task_name]
            required_args = getattr(interface_cls, 'REQUIRED_ARGS', [])
            default_args = getattr(interface_cls, 'DEFAULT_VARS', {})
            aliases = getattr(interface_cls, 'ALIASES', {}) 
            
            if required_args:
                print("\n[Required Arguments]")
                for arg in required_args:
                    prev_val = self._get_known_arg(arg, aliases)
                    if prev_val is not None:
                        val = self._prompt(f"{arg}", default=prev_val)
                    else:
                        val = self._prompt(f"{arg}", required=True)
                        
                    task_info["arguments"][arg] = val
                    self.known_args[arg] = val 
            
            if default_args:
                print(f"\n[Default Arguments] This interface has {len(default_args)} default arguments.")
                print("Note: Global arguments take precedence over interface defaults automatically.")
                
                for arg, default_val in default_args.items():
                    if arg not in required_args:
                        global_val = self._get_known_arg(arg, aliases)
                        if global_val is not None:
                            task_info["arguments"][arg] = global_val
                        else:
                            task_info["arguments"][arg] = default_val

                override = input("Do you want to override any default arguments manually? (y/N): ").strip().lower()
                
                if override == 'y':
                    for arg, default_val in default_args.items():
                        if arg in required_args: continue 
                        suggested_default = task_info["arguments"][arg]
                        val = self._prompt(f"{arg}", default=suggested_default)
                        task_info["arguments"][arg] = val
                        self.known_args[arg] = val  
                else:
                    print("  -> Global arguments will be applied, and any defaults with no global args will be applied.")
                    for arg in task_info["arguments"]:
                        if arg not in self.known_args:
                            self.known_args[arg] = task_info["arguments"][arg]

            if omit_globals:
                keys_to_remove = []
                for arg, val in task_info["arguments"].items():
                    if self._matches_global(arg, val, global_args, aliases):
                        keys_to_remove.append(arg)
                
                if keys_to_remove:
                    print(f"  -> Omitting {len(keys_to_remove)} redundant arguments from {task_key} to save space.")
                    for k in keys_to_remove:
                        del task_info["arguments"][k]

            if phase == "testing":
                # Robustly find the dataset path (check task args first, then globals)
                ds_keys = ["dataset", "data", "input", "test_dataset"]
                dataset_path = next((task_info["arguments"].get(k) for k in ds_keys if k in task_info["arguments"]), global_args.get("test_dataset"))
                
                eval_records.append({
                    "task_name": task_key,
                    "save_dir": task_info["arguments"].get("save_dir"),
                    "input_dir": dataset_path 
                })

            tasks[task_key] = task_info
            print(f"\nSuccessfully added {task_key} to job queue.")
            
        return tasks, eval_records
        
    def _configure_phase(self, phase):
        global_args = self.build_global_args(phase)

        print("\n" + "="*50)
        print(f"JSON Optimization [{phase.upper()}]")
        print("="*50)
        print("You can automatically omit task arguments from the JSON if they are")
        print("identical to the global arguments. This keeps the JSON file much smaller.")
        omit_globals = input("Do you want to omit matching arguments from tasks? (y/N): ").strip().lower() == 'y'

        tasks_result = self.build_tasks(phase, global_args, omit_globals)

        # unpack
        if phase == "testing":
            tasks, eval_records = tasks_result
            
            # --- [NEW] INPUT FOR EVALUATION ENVIRONMENT ---
            print("\n" + "="*50)
            print("Configuring Evaluation Environment")
            print("="*50)
            eval_env = self._prompt("Enter evaluation environment path", default="LIC-Models/eval-env", is_path=True)

            evaluation = {
                "env_path": eval_env,
                "tasks": eval_records
            }

            return {
                "global_arguments": global_args,
                "tasks": tasks,
                "evaluation": evaluation
            }

        else:
            tasks, _ = tasks_result
            return {
                "global_arguments": global_args,
                "tasks": tasks
            }

    def generate(self, output_path="arguments.json"):
        final_config = {}
        
        if self.train_mode:
            final_config["training"] = self._configure_phase("training")
        
        if self.test_mode:
            final_config["testing"] = self._configure_phase("testing")
        
        with open(output_path, 'w') as f:
            json.dump(final_config, f, indent=4)
            
        print("\n" + "="*50)
        print(f"Successfully generated {output_path}")
        print("="*50)


def parse_args():
    parser = argparse.ArgumentParser(description="Interactive Argument JSON Generator")
    parser.add_argument('--train', action='store_true', help="Enable training phase configuration")
    parser.add_argument('--test', action='store_true', help="Enable testing phase configuration")
    parser.add_argument('--train_path', type=str, default="Interfaces/Training-Interfaces", help="Path to training interfaces")
    parser.add_argument('--test_path', type=str, default="Interfaces/Testing-Interfaces", help="Path to testing interfaces")
    parser.add_argument('--output', type=str, default="arguments.json", help="Path for the generated JSON file")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # --- INTERACTIVE MENU ---
    if not args.train and not args.test:
        print("\nNo configuration phase selected. Please select an option:")
        print("  1) Configure Train and Evaluation jobs")
        print("  2) Configure Training jobs only")
        print("  3) Configure Evaluation jobs only")
        
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
        print() 

    # --- DIRECTORY VERIFICATION ---
    if args.train and not os.path.isdir(args.train_path):
        print(f"FATAL: You requested training configuration, but the directory '{args.train_path}' does not exist.")
        sys.exit(1)
        
    if args.test and not os.path.isdir(args.test_path):
        print(f"FATAL: You requested testing configuration, but the directory '{args.test_path}' does not exist.")
        sys.exit(1)
    
    # --- GENERATOR INITIALIZATION ---
    generator = ConfigGenerator(
        train_interfaces_path=args.train_path if args.train else None,
        test_interfaces_path=args.test_path if args.test else None,
        train_mode=args.train,
        test_mode=args.test
    )
    
    generator.generate(output_path=args.output)