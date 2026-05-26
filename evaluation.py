import os
import sys
import json
import argparse
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from piq import psnr 
from skimage.metrics import structural_similarity as ssim
import lpips

def get_bpp(bitstream_path, width, height):
    if not os.path.exists(bitstream_path):
        return 0.0
    file_size_bits = os.path.getsize(bitstream_path) * 8
    return file_size_bits / (width * height)

def rgb_to_yuv(tensor):
    # BT.601 conversion coefficients
    r, g, b = tensor[:, 0:1, :, :], tensor[:, 1:2, :, :], tensor[:, 2:3, :, :]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    u = -0.1687 * r - 0.3313 * g + 0.5 * b + 0.5
    v = 0.5 * r - 0.4187 * g - 0.0813 * b + 0.5
    return torch.cat((y, u, v), dim=1)

def main():
    # 1. Setup Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--input_dir", type=str, required=True)
    args = parser.parse_args()

    # 2. Define Device FIRST
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 3. Initialize Models
    loss_fn_alex = lpips.LPIPS(net='alex')
    loss_fn_alex.to(device)

    print(f"DEBUG: checking save_dir {args.save_dir}")

    # 4. Set up paths adaptively
    base_save_path = os.path.expanduser(args.save_dir)
    
    # Search for reconstruction directory recursively
    recon_dir = None
    for root, dirs, files in os.walk(base_save_path):
        if "reconstruction" in dirs:
            recon_dir = os.path.join(root, "reconstruction")
            break
        elif "reconstructions" in dirs:
            recon_dir = os.path.join(root, "reconstructions")
            break
            
    if not recon_dir:
        print(f"Error: Neither 'reconstructions' nor 'reconstruction' directory found in {base_save_path}")
        return

    # Derive ssim_map and psnr_map dirs from recon_dir parent
    ssim_dir = os.path.join(os.path.dirname(recon_dir), "ssim_map")
    psnr_map_dir = os.path.join(os.path.dirname(recon_dir), "psnr_map")
    os.makedirs(ssim_dir, exist_ok=True)
    os.makedirs(psnr_map_dir, exist_ok=True)

    bits_dir = None
    for root, dirs, files in os.walk(base_save_path):
        if "bitstreams" in dirs:
            bits_dir = os.path.join(root, "bitstreams")
            break
        elif "bitstream" in dirs:
            bits_dir = os.path.join(root, "bitstream")
            break
    
    if not bits_dir:
        bits_dir = os.path.join(base_save_path, "bitstreams") # Fallback

    # Accept any standard output image extension safely
    valid_extensions = (".png", ".jpg", ".jpeg", ".webp")
    recon_files = sorted([f for f in os.listdir(recon_dir) if f.lower().endswith(valid_extensions)])
    
    metrics = {"psnr": [], "psnr_y": [], "psnr_u": [], "psnr_v": [], "ssim": [], "lpips": [], "bpp": []}
    per_image_results = []

    print(f"Evaluating {len(recon_files)} images for {args.task_name} on {device}...")

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

        gt_img = Image.open(gt_path).convert("RGB")
        rec_img = Image.open(os.path.join(recon_dir, r_file)).convert("RGB")

        gt = to_tensor(gt_img).unsqueeze(0).to(device)
        rec = to_tensor(rec_img).unsqueeze(0).to(device)

        # Ensure shapes match perfectly
        h, w = min(gt.size(2), rec.size(2)), min(gt.size(3), rec.size(3))
        gt, rec = gt[:, :, :h, :w], rec[:, :, :h, :w]

        # Calculate RGB Metrics
        psnr_val = psnr(rec, gt, data_range=1.0).item()
        lpips_val = loss_fn_alex(rec * 2 - 1, gt * 2 - 1).item()
        
        # SSIM with Map (scikit-image)
        # Move to CPU/Numpy for skimage
        gt_np = gt.squeeze().permute(1, 2, 0).cpu().numpy()
        rec_np = rec.squeeze().permute(1, 2, 0).cpu().numpy()
        
        ssim_val, ssim_map = ssim(gt_np, rec_np, data_range=1.0, channel_axis=2, full=True)
        
        # Save SSIM Map as image
        # Map values are -1 to 1. Map to 0-255 for visualization.
        # High value (1.0) = white (similarity), Low = black (difference)
        # ssim_map is [H, W, C]. We take mean across channels or just Y.
        if len(ssim_map.shape) == 3:
            ssim_map_gray = np.mean(ssim_map, axis=2)
        else:
            ssim_map_gray = ssim_map
            
        ssim_map_img = Image.fromarray((np.clip(ssim_map_gray, 0, 1) * 255).astype(np.uint8))
        ssim_map_img.save(os.path.join(ssim_dir, f"{base_no_ext}.png"))
        
        # Calculate PSNR Map (per-pixel MSE)
        # Scale range to 0-1 based on the maximum error in THIS image
        mse_map = (gt_np - rec_np) ** 2
        if len(mse_map.shape) == 3:
            mse_map_gray = np.mean(mse_map, axis=2)
        else:
            mse_map_gray = mse_map
            
        max_val = np.max(mse_map_gray)
        if max_val > 0:
            mse_map_gray = mse_map_gray / max_val
            
        psnr_map_img = Image.fromarray((np.clip(mse_map_gray, 0, 1) * 255).astype(np.uint8))
        psnr_map_img.save(os.path.join(psnr_map_dir, f"{base_no_ext}.png"))
        
        # Calculate YUV Metrics
        gt_yuv = rgb_to_yuv(gt)
        rec_yuv = rgb_to_yuv(rec)
        
        psnr_y = psnr(rec_yuv[:, 0:1, :, :], gt_yuv[:, 0:1, :, :], data_range=1.0).item()
        psnr_u = psnr(rec_yuv[:, 1:2, :, :], gt_yuv[:, 1:2, :, :], data_range=1.0).item()
        psnr_v = psnr(rec_yuv[:, 2:3, :, :], gt_yuv[:, 2:3, :, :], data_range=1.0).item()

        # Adaptive search for bitstream files
        bits_file = None
        for b_cand in [f"{base_no_ext}.pt", f"bits_{base_no_ext}.pt", f"{base_no_ext}.bin", f"bits_{base_no_ext}.bin"]:
            candidate_path = os.path.join(bits_dir, b_cand)
            if os.path.exists(candidate_path):
                bits_file = candidate_path
                break
                
        bpp_val = get_bpp(bits_file, w, h) if bits_file else 0.0

        # Append to raw data pools for rolling average calculation
        metrics["psnr"].append(psnr_val)
        metrics["psnr_y"].append(psnr_y)
        metrics["psnr_u"].append(psnr_u)
        metrics["psnr_v"].append(psnr_v)
        metrics["ssim"].append(ssim_val)
        metrics["lpips"].append(lpips_val)
        metrics["bpp"].append(bpp_val)

        # Save granular data for individual tracking profile
        per_image_results.append({
            "image_name": r_file,
            "psnr": round(psnr_val, 4),
            "psnr_y": round(psnr_y, 4),
            "psnr_u": round(psnr_u, 4),
            "psnr_v": round(psnr_v, 4),
            "ssim": round(float(ssim_val), 4),
            "lpips": round(lpips_val, 4),
            "bpp": round(bpp_val, 4)
        })

    # 6. Compute Averages and Print results
    averages = {}
    print(f"\n--- Final Results for {args.task_name} ---")
    for k, v in metrics.items():
        if v:
            avg_val = float(sum(v) / len(v))
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