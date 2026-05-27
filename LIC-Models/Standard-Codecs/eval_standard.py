import os
import sys
import subprocess
import argparse
import time
from pathlib import Path

def run_command(cmd):
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {' '.join(cmd)}")
        print(f"Stderr: {e.stderr}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Evaluate standard video codecs on images.")
    parser.add_argument("--codec", type=str, required=True, choices=["AVC", "HEVC", "VVC", "AV1"])
    parser.add_argument("--qp", type=int, default=23)
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU-based encoder (NVENC) if available.")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    save_dir = Path(args.save_dir)
    
    recon_dir = save_dir / "reconstruction"
    bits_dir = save_dir / "bitstreams"
    
    recon_dir.mkdir(parents=True, exist_ok=True)
    bits_dir.mkdir(parents=True, exist_ok=True)

    # Map internal codec names to FFmpeg encoders and extensions
    if args.use_gpu:
        codec_map = {
            "AVC": {"encoder": "h264_nvenc", "ext": "h264", "args": ["-qp", str(args.qp), "-pix_fmt", "yuv420p"]},
            "HEVC": {"encoder": "hevc_nvenc", "ext": "hevc", "args": ["-qp", str(args.qp), "-pix_fmt", "yuv420p"]},
            "AV1": {"encoder": "av1_nvenc", "ext": "ivf", "args": ["-qp", str(args.qp), "-pix_fmt", "yuv420p"]},
            "VVC": {"encoder": "libvvenc", "ext": "266", "args": ["-qp", str(args.qp), "-pix_fmt", "yuv420p"]} # Fallback to SW
        }
    else:
        codec_map = {
            "AVC": {"encoder": "libx264", "ext": "h264", "args": ["-crf", str(args.qp), "-pix_fmt", "yuv420p"]},
            "HEVC": {"encoder": "libx265", "ext": "hevc", "args": ["-crf", str(args.qp), "-pix_fmt", "yuv420p", "-x265-params", "log-level=0"]},
            "AV1": {"encoder": "libsvtav1", "ext": "ivf", "args": ["-rc", "0", "-qp", str(args.qp), "-pix_fmt", "yuv420p"]},
            "VVC": {"encoder": "libvvenc", "ext": "266", "args": ["-qp", str(args.qp), "-pix_fmt", "yuv420p"]}
        }

    config = codec_map[args.codec]
    
    # Check if encoder exists
    check_enc = subprocess.run(["ffmpeg", "-encoders"], capture_output=True, text=True)
    if config["encoder"] not in check_enc.stdout:
        print(f"Error: Encoder {config['encoder']} for {args.codec} is not supported by your FFmpeg build.")
        print("Please install the necessary libraries or use a different FFmpeg build.")
        return

    valid_exts = [".png", ".jpg", ".jpeg", ".bmp"]
    image_files = sorted([f for f in input_dir.iterdir() if f.suffix.lower() in valid_exts])

    if not image_files:
        print(f"No images found in {args.input_dir}")
        return

    print(f"Encoding {len(image_files)} images using {args.codec} (QP={args.qp})...")

    for img_path in image_files:
        base_name = img_path.stem
        bitstream_path = bits_dir / f"{base_name}.{config['ext']}"
        reconstruction_path = recon_dir / f"{base_name}.png"

        # 1. Encode
        enc_cmd = [
            "ffmpeg", "-y", "-i", str(img_path),
            "-c:v", config["encoder"]
        ] + config["args"] + [
            "-f", config["ext"], str(bitstream_path)
        ]
        
        # Special case for AV1/SVT-AV1 which might need different flags depending on version
        if args.codec == "AV1" and config["encoder"] == "libsvtav1":
             # Try first with -qp, if it fails we could fallback to -crf but let's stick to one
             pass

        if not run_command(enc_cmd):
            print(f"Failed to encode {img_path.name}")
            continue

        # 2. Decode
        dec_cmd = [
            "ffmpeg", "-y", "-i", str(bitstream_path),
            str(reconstruction_path)
        ]
        
        if not run_command(dec_cmd):
            print(f"Failed to decode {bitstream_path.name}")
            continue

    print(f"Evaluation completed. Results saved in {save_dir}")

if __name__ == "__main__":
    main()
