import os
import subprocess
from base_interface import BaseInterface

class StableCodecTestInterface(BaseInterface):

    TASK_NAME = "StableCodec"
    
    USE_MODULE_EXECUTION = False
    # Updated to your new script location
    EXECUTION_PATH = "src/compress.py" 
    
    # Added codec_path to requirements
    REQUIRED_ARGS = ["img_path", "rec_path", "bin_path", "sd_path", "elic_path", "codec_path"]
    
    ACTION_FLAGS = [
        "color_fix",
        "enable_xformers_memory_efficient_attention"
    ]

    DEFAULT_VARS = {
        "sd_path": None,
        "elic_path": None,
        "codec_path": None,
        "img_path": None,
        "rec_path": "results/reconstructions",
        "bin_path": "results/binaries",
        "seed": 100,
        "color_fix": False,
        "enable_xformers_memory_efficient_attention": False
    }

    ALIASES = {
        "input": "img_path",
        "output": "rec_path",
        "bin": "bin_path",
        "codec": "codec_path"
    }

    CLI_MAPPING = {
        "sd_path": "--sd_path",
        "elic_path": "--elic_path",
        "codec_path": "--codec_path",
        "seed": "--seed",
        "img_path": "--img_path",
        "rec_path": "--rec_path",
        "bin_path": "--bin_path",
        "color_fix": "--color_fix",
        "enable_xformers_memory_efficient_attention": "--enable_xformers_memory_efficient_attention"
    }

    def __init__(self, job_args=None, global_args=None):
        super().__init__(job_args, global_args)
        
        # Ensure directories exist
        for key in ["rec_path", "bin_path"]:
            if key in self.params and self.params[key]:
                os.makedirs(self.params[key], exist_ok=True)
        
        # Internal cleanup
        if "cuda" in self.params:
            del self.params["cuda"]

    def execute(self):
        # We assume dependencies are handled by the environment 
        # or checked by your training interface for this project
        super().execute()