import subprocess
import os
import platform

class BaseInterface:
    """
    The Base Interface handles validation and command construction dynamically.

    Subclasses define model-specific configuration (required args, CLI mappings,
    execution paths, etc.) and this base class takes care of merging parameters,
    building the subprocess command, and executing it with the correct environment.
    """

    # --- Subclass configuration (override in each Testing-Interface) -----------

    REQUIRED_ARGS = []          # Keys that must be present for the job to run
    DEFAULT_VARS = {}           # Fallback values applied before any user args
    ALIASES = {}                # Maps alternate arg names -> canonical names
    CLI_MAPPING = {}            # Maps canonical arg names -> CLI flags (e.g. "input_dir": "-i")
    ACTION_FLAGS = []           # Boolean switches that take no trailing value (e.g. --cuda)
    EXECUTION_PATH = ""         # Script path or module name to run

    WORKING_DIR = None          # If set, subprocess cwd is resolved relative to this
    USE_MODULE_EXECUTION = False  # If True, run as `python -m <module>` instead of `python <script>`

    # --------------------------------------------------------------------------

    def __init__(self, job_args=None, global_args=None):
        """
        Merge parameters from three layers (lowest to highest priority):
          1. DEFAULT_VARS  - hardcoded defaults from the subclass
          2. global_args   - shared settings passed to every job (e.g. input_dir)
          3. job_args      - per-job overrides (e.g. quality level, checkpoint)

        Alias resolution is applied to both global_args and job_args so callers
        can use either the friendly name or the canonical name.
        """
        job_args = job_args or {}
        global_args = global_args or {}

        # Start from defaults, then layer on global and job-specific args
        self.params = self.DEFAULT_VARS.copy()

        clean_job_args = {self.ALIASES.get(k, k): v for k, v in job_args.items()}
        clean_global_args = {self.ALIASES.get(k, k): v for k, v in global_args.items()}

        self.params.update(clean_global_args)    
        self.params.update(clean_job_args)

        # Coerce string "true"/"false" to actual booleans so ACTION_FLAGS work
        for key, value in self.params.items():
            if isinstance(value, str):
                cleaned_val = value.strip().lower()
                if cleaned_val == "true":
                    self.params[key] = True
                elif cleaned_val == "false":
                    self.params[key] = False

    def validate(self):
        """
        Check that all REQUIRED_ARGS are present and non-None.
        Returns (True, []) on success or (False, [missing_keys]) on failure.
        """
        missing = [req for req in self.REQUIRED_ARGS if self.params.get(req) is None]
        if missing:
            return False, missing
        return True, []

    def build_command(self):
        """
        Construct the full subprocess command list.

        1. Resolve the Python executable:
           - If ENV_PATH points to a virtual environment, use its python binary.
           - Otherwise fall back to the system python/python3.
        2. Append the script path or module name (EXECUTION_PATH).
        3. Iterate CLI_MAPPING to translate canonical params into CLI flags.
        """

        # --- Resolve the Python executable from the venv -----------------------
        if hasattr(self, 'ENV_PATH') and self.ENV_PATH:
            # Try the path as-is first, then relative to WORKING_DIR
            if os.path.exists(os.path.abspath(self.ENV_PATH)):
                abs_env = os.path.abspath(self.ENV_PATH)
            elif getattr(self, 'WORKING_DIR', None) and os.path.exists(os.path.abspath(os.path.join(self.WORKING_DIR, self.ENV_PATH))):
                abs_env = os.path.abspath(os.path.join(self.WORKING_DIR, self.ENV_PATH))
            else:
                abs_env = os.path.abspath(self.ENV_PATH)

            # Windows venvs store the binary in Scripts/, Linux/macOS in bin/
            if os.path.isdir(os.path.join(abs_env, "Scripts")):
                python_executable = os.path.join(abs_env, "Scripts", "python.exe")
            else:
                python_executable = os.path.join(abs_env, "bin", "python3")
                
            command = [python_executable]
        else:
            # No venv specified -- use the system interpreter
            command = ["python"] if platform.system() == "Windows" else ["python3"]        

        # --- Append the target script or module --------------------------------
        if self.USE_MODULE_EXECUTION:
            command.extend(["-m", self.EXECUTION_PATH])
        else:
            command.append(self.EXECUTION_PATH)

        # --- Build CLI arguments from params via CLI_MAPPING -------------------
        for standard_key, cli_flag in self.CLI_MAPPING.items():
            if standard_key in self.params:
                val = self.params[standard_key]
                if val is None or val == "":
                    continue

                # Boolean switches (e.g. --cuda, --real): append flag only, no value
                if standard_key in getattr(self, 'ACTION_FLAGS', []):
                    if val is True: 
                        command.append(cli_flag)
                        
                # List/tuple values: append flag followed by each element
                elif isinstance(val, (list, tuple)):
                    command.append(cli_flag)
                    command.extend([str(v) for v in val])
                    
                # Standard key-value pairs: append flag and its value
                else:
                    command.append(cli_flag)
                    command.append(str(val))
                    
        return command
    
    def execute(self):
        """
        Run the constructed command as a subprocess.

        Environment setup:
          - PATH: Prepends the venv bin/ directory and the pip-installed CUDA
            toolkit's bin/ directory so that tools like `ninja` and `nvcc` are
            discoverable. The CUDA toolkit directory is discovered dynamically
            by scanning for nvidia/cuNN/ inside the venv's site-packages. This
            is required for models that JIT-compile CUDA C++ extensions at
            runtime (e.g. RwkvCompress's biwkv4, DCVC-RT's rans, HPCM's
            unbounded_rans).
          - CUDA_HOME: If not already set, points to the pip-installed CUDA
            toolkit or falls back to /usr/local/cuda. PyTorch's
            cpp_extension.load() needs this to locate CUDA headers and libs.
          - PYTHONPATH: Prepends the working directory so that relative imports
            within the model's codebase resolve correctly.
        """
        command = self.build_command()
        print(f"Executing:\n{' '.join(command)}\n")
        
        run_dir = os.path.abspath(self.WORKING_DIR) if self.WORKING_DIR else os.getcwd()
        
        run_env = os.environ.copy()

        # --- CUDA / JIT compilation environment setup --------------------------
        # When using a venv Python binary, the subprocess inherits the parent's
        # PATH which may not include the venv's bin/ (where ninja lives) or the
        # pip-installed CUDA toolkit's bin/ (where nvcc lives). Without these,
        # models that JIT-compile C++ extensions at runtime will fail with errors
        # like "Ninja is required to load C++ extensions".
        if command and os.path.exists(command[0]):
            bin_dir = os.path.dirname(os.path.abspath(command[0]))

            # Prepend the venv's bin/ so tools like ninja are discoverable
            run_env["PATH"] = f"{bin_dir}{os.pathsep}{run_env.get('PATH', '')}"

            # Discover the venv's site-packages path dynamically.
            # We scan the venv's lib/ for the actual pythonX.Y directory rather
            # than using sys.version_info, because the dispatcher's Python version
            # may differ from the venv's Python version.
            venv_root = os.path.abspath(os.path.join(bin_dir, ".."))
            lib_dir = os.path.join(venv_root, "lib")
            site_packages = None
            if os.path.isdir(lib_dir):
                python_dirs = sorted(
                    [d for d in os.listdir(lib_dir) if d.startswith("python")],
                    reverse=True  # prefer highest version if multiple exist
                )
                if python_dirs:
                    site_packages = os.path.join(lib_dir, python_dirs[0], "site-packages")

            # If we found site-packages, look for a pip-installed CUDA toolkit.
            # nvidia-cuda-runtime-cu13 installs to nvidia/cu13/, cu12 to nvidia/cu12/, etc.
            # The nvidia/ directory also contains unrelated packages (cudnn, cuda_runtime,
            # cusparselt, etc.), so we match only directories named "cu" + digits.
            if site_packages:
                nvidia_dir = os.path.join(site_packages, "nvidia")
                cuda_toolkit = None
                if os.path.isdir(nvidia_dir):
                    cu_dirs = sorted(
                        [d for d in os.listdir(nvidia_dir)
                         if d.startswith("cu") and d[2:].isdigit()],
                        reverse=True  # prefer highest version (e.g. cu13 over cu12)
                    )
                    if cu_dirs:
                        cuda_toolkit = os.path.join(nvidia_dir, cu_dirs[0])

                if cuda_toolkit:
                    # Prepend the CUDA toolkit's bin/ so nvcc is found
                    nvcc_bin = os.path.join(cuda_toolkit, "bin")
                    if os.path.isdir(nvcc_bin):
                        run_env["PATH"] = f"{nvcc_bin}{os.pathsep}{run_env['PATH']}"

                    # Set CUDA_HOME so PyTorch's JIT compiler can find headers/libs
                    if "CUDA_HOME" not in run_env:
                        run_env["CUDA_HOME"] = cuda_toolkit

            # Fallback: use system CUDA if no pip-installed toolkit was found
            if "CUDA_HOME" not in run_env and os.path.exists("/usr/local/cuda"):
                run_env["CUDA_HOME"] = "/usr/local/cuda"

        # --- PYTHONPATH setup --------------------------------------------------
        # Prepend the working directory so relative imports within the model's
        # codebase resolve correctly when running from a different cwd.
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