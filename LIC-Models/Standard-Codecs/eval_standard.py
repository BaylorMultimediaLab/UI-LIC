import os
import sys
import subprocess
import argparse
import time
import json
from pathlib import Path
from PIL import Image

def log(msg):
    print(msg, flush=True)

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
    parser.add_argument("--codec", type=str, required=True, help="Codec name or comma-separated list for equalization (e.g. AVC,HEVC,AV1)")
    parser.add_argument("--qp", type=int, default=23)
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU-based encoder (NVENC) if available.")
    parser.add_argument("--equalize", action="store_true", help="Perform 2-stage equalization to match bitrate of most efficient codec.")
    parser.add_argument("--target_bpp_json", type=str, default=None, help="Optional per-image target bpp map (JSON).")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    save_dir = Path(args.save_dir)
    
    # Support for multiple codecs in one run (needed for equalization)
    selected_codecs = args.codec.split(',')
    codec_save_dirs = {}
    qp_maps = {c: {} for c in selected_codecs}
    
    # Map internal codec names to FFmpeg encoders and extensions
    def get_codec_config(codec_name, qp, use_gpu):
        if use_gpu:
            cmap = {
                "AVC": {"encoder": "h264_nvenc", "ext": "h264", "args": ["-qp", str(qp), "-pix_fmt", "yuv420p"]},
                "HEVC": {"encoder": "hevc_nvenc", "ext": "hevc", "args": ["-qp", str(qp), "-pix_fmt", "yuv420p"]},
                "AV1": {"encoder": "av1_nvenc", "ext": "ivf", "args": ["-qp", str(qp), "-pix_fmt", "yuv420p"]}
            }
        else:
            cmap = {
                "AVC": {"encoder": "libx264", "ext": "h264", "args": ["-crf", str(qp), "-pix_fmt", "yuv420p"]},
                "HEVC": {"encoder": "libx265", "ext": "hevc", "args": ["-crf", str(qp), "-pix_fmt", "yuv420p", "-x265-params", "log-level=0"]},
                "AV1": {"encoder": "libsvtav1", "ext": "ivf", "args": ["-rc", "0", "-qp", str(qp), "-pix_fmt", "yuv420p"]}
            }
        return cmap.get(codec_name)

    # Check encoders availability
    check_enc = subprocess.run(["ffmpeg", "-encoders"], capture_output=True, text=True)
    for c in selected_codecs:
        cfg = get_codec_config(c, args.qp, args.use_gpu)
        if not cfg or cfg["encoder"] not in check_enc.stdout:
            print(f"Error: Encoder {cfg['encoder'] if cfg else 'Unknown'} for {c} is not supported.")
            return

    valid_exts = [".png", ".jpg", ".jpeg", ".bmp"]
    image_files = sorted([f for f in input_dir.iterdir() if f.suffix.lower() in valid_exts])

    if not image_files:
        print(f"No images found in {args.input_dir}")
        return

    log(f"Evaluating {len(image_files)} images using {args.codec} (QP={args.qp}, Equalize={args.equalize})...")

    qp_bounds = {
        "AVC": (0, 51),
        "HEVC": (0, 51),
        "AV1": (0, 255)
    }

    coarse_step = 7
    fine_step = 1

    target_bpp_map = {}
    if args.target_bpp_json:
        try:
            with open(args.target_bpp_json, "r") as f:
                target_bpp_map = json.load(f)
        except Exception as e:
            log(f"[WARNING] Failed to load target bpp map: {e}")
            target_bpp_map = {}

    for img_path in image_files:
        base_name = img_path.stem
        with Image.open(img_path) as img:
            w, h = img.size
        
        # Determine target bpp per image if equalizing
        results = {} # codec -> {bpp, path}

        # Stage 1: Initial encoding at base QP
        for cname in selected_codecs:
            # We must save results into codec-specific folders inside save_dir
            c_save_dir = save_dir if len(selected_codecs) == 1 else save_dir.parent / cname
            codec_save_dirs[cname] = c_save_dir
            c_recon_dir = c_save_dir / "reconstruction"
            c_bits_dir = c_save_dir / "bitstreams"
            c_recon_dir.mkdir(parents=True, exist_ok=True)
            c_bits_dir.mkdir(parents=True, exist_ok=True)
            
            cfg = get_codec_config(cname, args.qp, args.use_gpu)
            bit_path = c_bits_dir / f"{base_name}.{cfg['ext']}"
            rec_path = c_recon_dir / f"{base_name}.png"
            
            # Encode
            enc_cmd = ["ffmpeg", "-y", "-i", str(img_path), "-c:v", cfg["encoder"]] + cfg["args"] + ["-f", cfg["ext"], str(bit_path)]
            if run_command(enc_cmd):
                # Calc bpp
                bpp = (os.path.getsize(bit_path) * 8) / (w * h)
                results[cname] = {"bpp": bpp, "qp": args.qp, "bit_path": bit_path, "rec_path": rec_path}
            
        if args.equalize and len(results) >= 1:
            # Stage 2: Equalization
            # Determine target bpp from learning-based models if available.
            # If not available, fall back to the most efficient standard codec as anchor.
            
            learning_target = target_bpp_map.get(base_name)
            
            anchor_codec = min(results, key=lambda k: results[k]["bpp"])
            standard_anchor_bpp = results[anchor_codec]["bpp"]
            
            if isinstance(learning_target, (int, float)) and learning_target > 0:
                # Requirement: The lowest BPP of learned codecs is the target rate.
                target_bpp = learning_target
                log(
                    f"  -> [{base_name}] Anchor: Learned codec at {target_bpp:.4f} bpp | "
                    f"Standard (for reference): {anchor_codec} at {standard_anchor_bpp:.4f} bpp"
                )
            else:
                target_bpp = standard_anchor_bpp
                log(f"  -> [{base_name}] Anchor: {anchor_codec} at {target_bpp:.4f} bpp")
            
            for cname in selected_codecs:
                # If we are already extremely close to target (e.g. this WAS the anchor and no lower target exists)
                if abs(results[cname]["bpp"] - target_bpp) < 0.00001:
                    log(f"  -> [{base_name}] {cname} already matches target bpp ({results[cname]['bpp']:.4f}).")
                    continue
                
                log(f"  -> [{base_name}] Equalizing {cname} to target {target_bpp:.4f}...")
                
                # Search for best QP to match target_bpp
                best_qp = results[cname]["qp"]
                best_diff = abs(results[cname]["bpp"] - target_bpp)

                # To match LOWER bpp, we need HIGHER QP.
                direction = 1 if results[cname]["bpp"] > target_bpp else -1

                min_qp, max_qp = qp_bounds.get(cname, (0, 51))
                curr_qp = args.qp
                attempts = 0
                max_attempts = max_qp - min_qp
                while attempts < max_attempts:
                    curr_qp += direction * coarse_step
                    if curr_qp < min_qp or curr_qp > max_qp:
                        break
                    attempts += coarse_step

                    cfg = get_codec_config(cname, curr_qp, args.use_gpu)
                    tmp_bit = results[cname]["bit_path"].with_suffix(".tmp")
                    enc_cmd = ["ffmpeg", "-y", "-i", str(img_path), "-c:v", cfg["encoder"]] + cfg["args"] + ["-f", cfg["ext"], str(tmp_bit)]

                    if run_command(enc_cmd):
                        bpp = (os.path.getsize(tmp_bit) * 8) / (w * h)
                        diff = abs(bpp - target_bpp)
                        if diff < best_diff:
                            best_diff = diff
                            best_qp = curr_qp
                            os.replace(tmp_bit, results[cname]["bit_path"])
                            results[cname]["bpp"] = bpp
                            results[cname]["qp"] = curr_qp
                        else:
                            if os.path.exists(tmp_bit):
                                os.remove(tmp_bit)
                    else:
                        break

                # Fine adjustment around best QP
                fine_min = max(min_qp, best_qp - coarse_step)
                fine_max = min(max_qp, best_qp + coarse_step)
                for curr_qp in range(fine_min, fine_max + 1, fine_step):
                    if curr_qp == best_qp:
                        continue
                    cfg = get_codec_config(cname, curr_qp, args.use_gpu)
                    tmp_bit = results[cname]["bit_path"].with_suffix(".tmp")
                    enc_cmd = ["ffmpeg", "-y", "-i", str(img_path), "-c:v", cfg["encoder"]] + cfg["args"] + ["-f", cfg["ext"], str(tmp_bit)]

                    if run_command(enc_cmd):
                        bpp = (os.path.getsize(tmp_bit) * 8) / (w * h)
                        diff = abs(bpp - target_bpp)
                        if diff < best_diff:
                            best_diff = diff
                            best_qp = curr_qp
                            os.replace(tmp_bit, results[cname]["bit_path"])
                            results[cname]["bpp"] = bpp
                            results[cname]["qp"] = curr_qp
                        else:
                            if os.path.exists(tmp_bit):
                                os.remove(tmp_bit)
                    else:
                        break
                log(f"    - {cname} equalized to QP {best_qp} ({results[cname]['bpp']:.4f} bpp)")

        # Final decoding for the chosen QPs
        for cname, res in results.items():
            dec_cmd = ["ffmpeg", "-y", "-i", str(res["bit_path"]), str(res["rec_path"])]
            run_command(dec_cmd)
            qp_maps[cname][base_name] = res.get("qp", args.qp)

    # Save per-image QP maps for downstream metrics
    for cname, qp_map in qp_maps.items():
        c_save_dir = codec_save_dirs.get(cname)
        if not c_save_dir:
            continue
        qp_path = c_save_dir / "qp_map.json"
        with open(qp_path, "w") as f:
            json.dump(qp_map, f, indent=4)

    log("Evaluation completed.")

if __name__ == "__main__":
    main()
