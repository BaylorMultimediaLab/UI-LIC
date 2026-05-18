import os
import sys
import json
import time
import argparse
import inspect
import importlib.util
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


        for step, (task_name, task_info) in enumerate(tasks.items(), start=1):
            # print("\n" + "="*50)
            # print(f"Step {step}/{len(job_queue)}: Starting [{task_name}]")
            # print("="*50)
            
            task_name = task_info.get("task_name")
            custom_dir = task_info.get("directory")
            
            env_path = task_info.get("env_path")
            job_args = task_info.get("arguments", {})

            # Check if the requested string matches a loaded Interface's task_name
            if task_name not in self.registry:
                print(f"  -> [SKIPPED] Unknown task type: '{task_name}'.")
                print(f"     (Make sure an interface file with TASK_NAME = '{task_name}' exists).")
                continue

            # 4. Instantiate the Interface
            # We pass the entire global pool. The interface's ALIASES will safely extract only what it needs.
            InterfaceClass = self.registry[task_name]
            interface_instance = InterfaceClass(job_args=job_args, global_args=global_args)
            
            # Override the Interface's default directory if one was provided in the JSON
            if custom_dir:
                interface_instance.WORKING_DIR = custom_dir
                
            if env_path:
                interface_instance.ENV_PATH = env_path
                
            is_valid, missing_args = interface_instance.validate()
            
            if not is_valid:
                print(f"  -> [FAILED] Interface '{task_name}' is missing required arguments: {missing_args}")
                print(f"  -> Skipping to next job...")
                continue

            print("  -> [VERIFIED] All required arguments present. Executing...\n")
            try:
                interface_instance.execute()
            except Exception as e:
                print(f"\n  -> [ERROR] Execution of '{task_name}' crashed: {e}")
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
