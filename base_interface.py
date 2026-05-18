import subprocess
import os

class BaseInterface:
    """
    The Base Interface handles validation and command construction dynamically.
    """
    REQUIRED_ARGS = []
    DEFAULT_VARS = {}
    ALIASES = {}
    CLI_MAPPING = {}
    EXECUTION_PATH = ""
    
    WORKING_DIR = None
    USE_MODULE_EXECUTION = False

    def __init__(self, job_args=None, global_args=None):
        job_args = job_args or {}
        global_args = global_args or {}
        
        self.params = self.DEFAULT_VARS.copy()
        
        clean_job_args = {self.ALIASES.get(k, k): v for k, v in job_args.items()}
        
        clean_global_args = {self.ALIASES.get(k, k): v for k, v in global_args.items()}

        # 4. Apply the hierarchy!
        self.params.update(clean_job_args)       # Local args overwrite defaults
        self.params.update(clean_global_args)    # Global args overwrite EVERYTHING

    def validate(self):
        missing = [req for req in self.REQUIRED_ARGS if self.params.get(req) is None]
        if missing:
            return False, missing
        return True, []

    def build_command(self):
        if hasattr(self, 'ENV_PATH') and self.ENV_PATH:
            # 1. Resolve the absolute path of the environment
            if self.WORKING_DIR:
                abs_env = os.path.abspath(os.path.join(self.WORKING_DIR, self.ENV_PATH))
            else:
                abs_env = os.path.abspath(self.ENV_PATH)
            
            # --- THE FIX: Point directly to the environment's python executable ---
            # (Note: For Windows compatibility in the future, it would be "Scripts\\python.exe")
            python_executable = os.path.join(abs_env, "bin", "python3")
            command = [python_executable]
            # ----------------------------------------------------------------------
        else:
            # Fallback to standard python if no env_path was in the JSON
            command = ["python3"]        
        
        # 2. Add the module or script path
        if self.USE_MODULE_EXECUTION:
            command.extend(["-m", self.EXECUTION_PATH])
        else:
            command.append(self.EXECUTION_PATH)

        # 3. Add all the CLI arguments
        for standard_key, cli_flag in self.CLI_MAPPING.items():
            if standard_key in self.params:
                val = self.params[standard_key]
                if val is None:
                    continue
                if isinstance(val, bool):
                    if val is True:
                        command.append(cli_flag)
                elif isinstance(val, (list, tuple)):
                    command.append(cli_flag)
                    command.extend([str(v) for v in val])
                else:
                    command.extend([cli_flag, str(val)])
                    
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