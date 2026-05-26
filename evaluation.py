import os
import sys
import json
import argparse
import torch
from PIL import Image
from torchvision import transforms
from piq import psnr, ssim 
import lpips

# Import Docker VMAF helper
try:
    from vmaf_docker import calculate_vmaf, check_docker_availability
except ImportError:
    calculate_vmaf = None
    check_docker_availability = None

def get_bpp(bitstream_path, width, height):
    if not os.path.exists(bitstream_path):
        return 0.0
    file_size_bits = os.path.getsize(bitstream_path) * 8
    return file_size_bits / (width * height)

def main():
    # 1. Setup Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--use_vmaf", action="store_true", help="Enable VMAF evaluation via Docker")
    args = parser.parse_args()

    # Safely handle VMAF dependency
    if args.use_vmaf:
        if calculate_vmaf is None or check_docker_availability is None:
            print("Warning: vmaf_docker.py not found. VMAF evaluation disabled.")
            args.use_vmaf = False
        else:
            is_available, msg = check_docker_availability()
            if not is_available:
                print(f"Warning: {msg}")
                print("VMAF evaluation will be skipped to prevent crashes.")
                args.use_vmaf = False
            else:
                print("VMAF check passed. Docker is ready.")

    # 2. Define Device FIRST
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 3. Initialize Models
    loss_fn_alex = lpips.LPIPS(net='alex')
    loss_fn_alex.to(device)

    print(f"DEBUG: checking save_dir {args.save_dir}")

    # 4. Set up paths adaptively
    base_save_path = os.path.expanduser(args.save_dir)
    
    recon_dir = os.path.join(base_save_path, "reconstruction")
        
    bits_dir = os.path.join(base_save_path, "bitstreams")
    if not os.path.exists(bits_dir):
        bits_dir = os.path.join(base_save_path, "bitstream")
    
    if not os.path.exists(recon_dir):
        print(f"Error: Neither 'reconstructions' nor 'reconstruction' directory found in {base_save_path}")
        return

    # Accept any standard output image extension safely
    valid_extensions = (".png", ".jpg", ".jpeg", ".webp")
    recon_files = sorted([f for f in os.listdir(recon_dir) if f.lower().endswith(valid_extensions)])
    
    metrics = {"psnr": [], "ssim": [], "lpips": [], "bpp": []}
    if args.use_vmaf:
        metrics["vmaf"] = []
        
    per_image_results = []

    print(f"Evaluating {len(recon_files)} images for {args.task_name} on {device}...")
    if args.use_vmaf:
        print("VMAF evaluation is ENABLED (using Docker). This may be slower.")

    # 5. Evaluation Loop
    to_tensor = transforms.ToTensor()
    for r_file in recon_files:
        clean_name = r_file[4:] if r_file.startswith("rec_") else r_file
        base_no_ext = os.path.splitext(clean_name)[0]
        
        # Match cleanly with ground truth dataset images
        gt_path = None
        for ext in [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"]:
            candidate = os.path.join(args.input_dir, base_no_ext + ext)
            if os.path.exists(candidate):
                gt_path = candidate
                break
        
        if not gt_path:
            continue

        recon_path = os.path.join(recon_dir, r_file)
        gt = to_tensor(Image.open(gt_path).convert("RGB")).unsqueeze(0).to(device)
        rec = to_tensor(Image.open(recon_path).convert("RGB")).unsqueeze(0).to(device)

        # Ensure shapes match perfectly
        h, w = min(gt.size(2), rec.size(2)), min(gt.size(3), rec.size(3))
        gt, rec = gt[:, :, :h, :w], rec[:, :, :h, :w]

        # Calculate Metrics
        psnr_val = psnr(rec, gt, data_range=1.0).item()
        ssim_val = ssim(rec, gt, data_range=1.0).item()
        lpips_val = loss_fn_alex(rec * 2 - 1, gt * 2 - 1).item()
        
        # Adaptive search for bitstream files
        bits_file = None
        for b_cand in [f"{base_no_ext}.pt", f"bits_{base_no_ext}.pt", f"{base_no_ext}.bin", f"bits_{base_no_ext}.bin"]:
            candidate_path = os.path.join(bits_dir, b_cand)
            if os.path.exists(candidate_path):
                bits_file = candidate_path
                break
                
        bpp_val = get_bpp(bits_file, w, h) if bits_file else 0.0

        # Calculate VMAF if requested
        vmaf_val = 0.0
        if args.use_vmaf:
            vmaf_val = calculate_vmaf(recon_path, gt_path)
            metrics["vmaf"].append(vmaf_val)

        # Append to raw data pools for rolling average calculation
        metrics["psnr"].append(psnr_val)
        metrics["ssim"].append(ssim_val)
        metrics["lpips"].append(lpips_val)
        metrics["bpp"].append(bpp_val)

        # Save granular data for individual tracking profile
        res_entry = {
            "image_name": r_file,
            "psnr": round(psnr_val, 4),
            "ssim": round(ssim_val, 4),
            "lpips": round(lpips_val, 4),
            "bpp": round(bpp_val, 4)
        }
        if args.use_vmaf:
            res_entry["vmaf"] = round(vmaf_val, 4)
            
        per_image_results.append(res_entry)

    # 6. Compute Averages and Print results
    averages = {}
    print(f"\n--- Final Results for {args.task_name} ---")
    for k, v in metrics.items():
        if v:
            avg_val = sum(v) / len(v)
            averages[k] = round(avg_val, 4)
            print(f"Average {k.upper()}: {avg_val:.4f}")
        else:
            averages[k] = 0.0

    # 7. Construct and Export JSON payload
    json_output = {
        "task_name": args.task_name,
        "averages": averages,
        "per_image_metrics": per_image_results
    }
    
    output_filename = f"{args.task_name}_metrics.json"
    output_filepath = os.path.join(base_save_path, output_filename)
    
    with open(output_filepath, "w") as f:
        json.dump(json_output, f, indent=4)
        
    print(f"SUCCESS: Metrics exported to {output_filepath}")

if __name__ == "__main__":
    main()