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

    def _prompt(self, message, default=None, required=False):
        """Helper to prompt the user for input."""
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
                    return default
                if required:
                    print("  -> This argument is required. Please provide a value.")
                    continue
                return None
            
            return self._parse_type(user_input)

    def build_global_args(self):
        print("\n" + "="*50)
        print("🔧 Configuring Global Arguments")
        print("="*50)
        
        global_args = {}
        
        # Standard global arguments template
        prompts = {
            "cuda": False,
            "gpu_id": 0,
            "epochs": 1,
            "learning_rate": 0.00067,
            "batch_size": 16,
            "test_batch_size": 1,
            "patch_size": "[256, 256]",
            "codestream_path": "",
            "train_dataset": "",
            "test_dataset": "",
            "train_split": "Positive",
            "test_split": "Negative"
        }
        
        for key, default_val in prompts.items():
            val = self._prompt(f"Enter {key}", default=default_val)
            if val is not None:
                global_args[key] = val
                self.known_args[key] = val  # Save to memory
                
        return global_args

    def build_tasks(self):
        print("\n" + "="*50)
        print("🚀 Configuring Tasks")
        print("="*50)
        
        if not self.registry:
            print("No interfaces found! Please ensure your interface directories are correct.")
            return {}
            
        print("Available Interfaces:")
        for task_name in self.registry.keys():
            print(f" - {task_name}")
            
        tasks = {}
        
        while True:
            task_name = input("\nEnter the Task Name to add (or press Enter to finish): ").strip()
            
            if not task_name:
                break
                
            if task_name not in self.registry:
                print(f"Unknown task type: '{task_name}'. Try again.")
                continue
                
            task_key = task_name
            existing_count = sum(1 for k in tasks.keys() if k.startswith(task_name))
            if existing_count > 0:
                task_key = f"{task_name}_{existing_count + 1}"
            
            print(f"\n--- Configuring Task: {task_key} ---")
            
            # 1. Base Task Information
            env_path = self._prompt("Enter env_path", default=f"~/{task_name}-env")
            directory = self._prompt("Enter directory", default=task_name)
            
            task_info = {
                "env_path": env_path,
                "task_name": task_name,
                "directory": directory,
                "arguments": {}
            }
            
            interface_cls = self.registry[task_name]
            required_args = getattr(interface_cls, 'REQUIRED_ARGS', [])
            default_args = getattr(interface_cls, 'DEFAULT_VARS', {}) 
            
            # 2. Required Arguments
            if required_args:
                print("\n[Required Arguments]")
                for arg in required_args:
                    # Check if we already know this value from global args or previous tasks
                    prev_val = self.known_args.get(arg)
                    
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
                
                # Pre-fill with defaults
                for arg, default_val in default_args.items():
                    task_info["arguments"][arg] = default_val
                    
                if override == 'y':
                    for arg, default_val in default_args.items():
                        # Skip if it was handled in required args
                        if arg in required_args:
                            continue 
                        
                        # Prioritize previously entered values over interface defaults if they exist
                        suggested_default = self.known_args.get(arg, default_val)
                        
                        val = self._prompt(f"{arg}", default=suggested_default)
                        task_info["arguments"][arg] = val
                        self.known_args[arg] = val  # Save to memory
                else:
                    # If user skips override, we still might want to add these to memory
                    for arg, default_val in default_args.items():
                        if arg not in self.known_args:
                            self.known_args[arg] = default_val

            tasks[task_key] = task_info
            print(f"✅ Added {task_key} to job queue.")
            
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
        print(f"🎉 Successfully generated {output_path}")
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