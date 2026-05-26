import subprocess
import os
import platform

class BaseInterface:
    """
    The Base Interface handles validation and command construction dynamically.
    """
    REQUIRED_ARGS = []
    DEFAULT_VARS = {}
    ALIASES = {}
    CLI_MAPPING = {}
    ACTION_FLAGS = []  # Defines which booleans are switches (no trailing values)
    EXECUTION_PATH = ""
    
    WORKING_DIR = None
    USE_MODULE_EXECUTION = False

    def __init__(self, job_args=None, global_args=None):
        job_args = job_args or {}
        global_args = global_args or {}
        
        self.params = self.DEFAULT_VARS.copy()
        
        clean_job_args = {self.ALIASES.get(k, k): v for k, v in job_args.items()}
        clean_global_args = {self.ALIASES.get(k, k): v for k, v in global_args.items()}

        self.params.update(clean_global_args)    
        self.params.update(clean_job_args)
        
        for key, value in self.params.items():
            if isinstance(value, str):
                cleaned_val = value.strip().lower()
                if cleaned_val == "true":
                    self.params[key] = True
                elif cleaned_val == "false":
                    self.params[key] = False

    def validate(self):
        missing = [req for req in self.REQUIRED_ARGS if self.params.get(req) is None]
        if missing:
            return False, missing
        return True, []

    def build_command(self):
        if hasattr(self, 'ENV_PATH') and self.ENV_PATH:
            if self.WORKING_DIR:
                abs_env = os.path.abspath(os.path.join(self.WORKING_DIR, self.ENV_PATH))
            else:
                abs_env = os.path.abspath(self.ENV_PATH)
            
            if os.path.isdir(os.path.join(abs_env, "Scripts")):
                python_executable = os.path.join(abs_env, "Scripts", "python.exe")
            else:
                python_executable = os.path.join(abs_env, "bin", "python3")
                
            command = [python_executable]
        else:
            command = ["python"] if platform.system() == "Windows" else ["python3"]        
        
        if self.USE_MODULE_EXECUTION:
            command.extend(["-m", self.EXECUTION_PATH])
        else:
            command.append(self.EXECUTION_PATH)

        # The single, unified argument builder loop
        for standard_key, cli_flag in self.CLI_MAPPING.items():
            if standard_key in self.params:
                val = self.params[standard_key]
                if val is None or val == "":
                    continue
                
                # Check if this flag is registered as a switch 
                if standard_key in getattr(self, 'ACTION_FLAGS', []):
                    if val is True: 
                        command.append(cli_flag)
                        
                # Handle lists/tuples
                elif isinstance(val, (list, tuple)):
                    command.append(cli_flag)
                    command.extend([str(v) for v in val])
                    
                # Handle standard key-values
                else:
                    command.append(cli_flag)
                    command.append(str(val))
                    
        return command
    
    def execute(self):
        command = self.build_command()
        print(f"Executing:\n{' '.join(command)}\n")
        
        run_dir = os.path.abspath(self.WORKING_DIR) if self.WORKING_DIR else os.getcwd()
        
        run_env = os.environ.copy()
        existing_pythonpath = run_env.get("PYTHONPATH", "")
        
        if existing_pythonpath:
            run_env["PYTHONPATH"] = f"{run_dir}{os.pathsep}{existing_pythonpath}"
        else:
            run_env["PYTHONPATH"] = run_dir
        
        try:
            subprocess.run(command, check=True, text=True, cwd=run_dir, env=run_env)
            print("[SUCCESS] Script completed.")
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Execution failed with code {e.returncode}")
            raise e