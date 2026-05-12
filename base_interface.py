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
        command = ["python3", self.EXECUTION_PATH]
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
        
        custom_env = os.environ.copy()
        
        if "PYTHONPATH" in custom_env:
            custom_env["PYTHONPATH"] = f"{run_dir}:{custom_env['PYTHONPATH']}"
        else:
            custom_env["PYTHONPATH"] = run_dir
        
        try:
            subprocess.run(command, check=True, text=True, cwd=run_dir, env=custom_env)
            print("[SUCCESS] Script completed.")
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Execution failed with code {e.returncode}")
            raise e