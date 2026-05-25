import os
from base_interface import BaseInterface


class StableCodecInterface(BaseInterface):

    TASK_NAME = "StableCodec"
    USE_MODULE_EXECUTION = False
    EXECUTION_PATH = "src/compress.py"

    # -----------------------------
    # ONLY TRUE REQUIRED ARGS
    # -----------------------------
    REQUIRED_ARGS = [
        "sd_path",
        "elic_path",
        "codec_path",
        "img_path",
        "save_dir",
    ]

    # -----------------------------
    # OPTIONAL FLAGS ONLY
    # -----------------------------
    ACTION_FLAGS = [
        "color_fix",
        "enable_xformers_memory_efficient_attention",
    ]

    # -----------------------------
    # INPUT CONTRACT
    # -----------------------------
    DEFAULT_VARS = {
        "sd_path": "LIC-Models/StableCodec/sd-turbo",
        "elic_path": "LIC-Models/StableCodec/elic.pth",
        "codec_path": None,
        "img_path": None,
        "save_dir": None,
    }

    # -----------------------------
    # CLI MAPPING
    # -----------------------------
    CLI_MAPPING = {
        "sd_path": "--sd_path",
        "elic_path": "--elic_path",
        "codec_path": "--codec_path",
        "img_path": "--img_path",

        # derived automatically
        "rec_path": "--rec_path",
        "bin_path": "--bin_path",

        "color_fix": "--color_fix",
        "enable_xformers_memory_efficient_attention":
            "--enable_xformers_memory_efficient_attention",
    }

    def __init__(self, job_args=None, global_args=None):
        super().__init__(job_args, global_args)

        # -----------------------------
        # SINGLE DIRECTORY CONTRACT
        # -----------------------------
        save_dir = self.params.get("save_dir")
        if not save_dir:
            raise ValueError("Missing save_dir")

        rec_path = os.path.join(save_dir, "reconstructions")
        bin_path = os.path.join(save_dir, "bitstreams")

        self.params["rec_path"] = rec_path
        self.params["bin_path"] = bin_path

        os.makedirs(rec_path, exist_ok=True)
        os.makedirs(bin_path, exist_ok=True)

        # -----------------------------
        # VALIDATION
        # -----------------------------
        missing = [
            k for k in self.REQUIRED_ARGS
            if not self.params.get(k)
        ]
        if missing:
            raise ValueError(f"Missing required StableCodec args: {missing}")

    def execute(self):
        super().execute()