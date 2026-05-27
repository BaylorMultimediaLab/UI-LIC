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
import scipy.ndimage

# Import Docker VMAF helper
try:
    from vmaf_docker import calculate_vmaf, check_docker_availability
except ImportError:
    calculate_vmaf = None
    check_docker_availability = None



def compute_color_gradient(img_np):
    if hasattr(img_np, "detach"):
        img_np = img_np.detach().cpu().numpy()

    if img_np.ndim == 4 and img_np.shape[0] == 1:
        img_np = img_np[0]

    if img_np.ndim == 3 and img_np.shape[0] == 3 and img_np.shape[-1] != 3:
        img_np = np.transpose(img_np, (1, 2, 0))

    if img_np.ndim == 3:
        img_np = img_np.mean(axis=2)

    grad_x = scipy.ndimage.sobel(img_np, axis=0, mode='reflect')
    grad_y = scipy.ndimage.sobel(img_np, axis=1, mode='reflect')

    grad_x_abs = np.abs(grad_x)
    grad_y_abs = np.abs(grad_y)

    grad_x_norm = grad_x_abs / (grad_x_abs.max() + 1e-8)
    grad_y_norm = grad_y_abs / (grad_y_abs.max() + 1e-8)
    grad_mag = np.sqrt(grad_x_abs**2 + grad_y_abs**2)
    grad_mag_norm = (grad_mag - grad_mag.min()) / (grad_mag.max() - grad_mag.min() + 1e-8)

    rgb = np.stack([grad_x_norm, grad_y_norm, grad_mag_norm], axis=-1)
    return (rgb * 255).astype(np.uint8)

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
    loss_fn_alex = lpips.LPIPS(net='alex',spatial=True).to(device)

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
    grad_dir = os.path.join(os.path.dirname(recon_dir), "grad_map")
    lpips_map_dir = os.path.join(os.path.dirname(recon_dir), "lpips_map")
    os.makedirs(ssim_dir, exist_ok=True)
    os.makedirs(psnr_map_dir, exist_ok=True)
    os.makedirs(grad_dir, exist_ok=True)
    os.makedirs(lpips_map_dir, exist_ok=True)

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
    if args.use_vmaf:
        metrics["vmaf"] = []
    per_image_results = []

    qp_map = {}
    qp_map_path = os.path.join(base_save_path, "qp_map.json")
    if os.path.exists(qp_map_path):
        try:
            with open(qp_map_path, "r") as f:
                qp_map = json.load(f)
        except Exception:
            qp_map = {}

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
       
        gt_img = Image.open(gt_path).convert("RGB")
        rec_img = Image.open(os.path.join(recon_dir, r_file)).convert("RGB")

        gt = to_tensor(gt_img).unsqueeze(0).to(device)
        rec = to_tensor(rec_img).unsqueeze(0).to(device)

        # Ensure shapes match perfectly
        h, w = min(gt.size(2), rec.size(2)), min(gt.size(3), rec.size(3))
        gt, rec = gt[:, :, :h, :w], rec[:, :, :h, :w]

        # Calculate RGB Metrics
        psnr_val = psnr(rec, gt, data_range=1.0).item()
        lpips_val, lpips_res = loss_fn_alex(rec, gt,retPerLayer=True, normalize=True)
        lpips_val = lpips_val.mean().item()
        # Stack into [5, 1, H, W], then sum or average over the first axis (layer)
        lpips_error_maps = torch.stack([r for r in lpips_res], dim=0)  # [5, 1, H, W]

        # Remove the channel dimension and average/sum over layers
        lpips_heatmap = lpips_error_maps.mean(dim=0)  # [1, H, W], or .sum(dim=0) for sum
        lpips_heatmap = lpips_heatmap.squeeze(0)      # [H, W]
        
        #debug print shape of lpips_res
        print(f"DEBUG: LPIPS layer outputs for {r_file}: {[res.shape for res in lpips_res]}")
        
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
        
        # Calculate and Save Color Gradient Map
        grad_map = compute_color_gradient(rec_np)
        grad_map_img = Image.fromarray(grad_map)
        grad_map_img.save(os.path.join(grad_dir, f"{base_no_ext}.png"))

        # Save LPIPS Heatmap
        heatmap_np = lpips_heatmap.detach().cpu().numpy()
        # Ensure 2D shape (H, W)
        if len(heatmap_np.shape) == 3:
             heatmap_np = heatmap_np.squeeze(0) # [1, H, W] -> [H, W]
        
        # Normalize for visualization
        h_min, h_max = heatmap_np.min(), heatmap_np.max()
        if h_max > h_min:
            heatmap_np = (heatmap_np - h_min) / (h_max - h_min)
            
        lpips_heatmap_img = Image.fromarray((np.clip(heatmap_np, 0, 1) * 255).astype(np.uint8))
        lpips_heatmap_img.save(os.path.join(lpips_map_dir, f"{base_no_ext}_lpips.png"))

        # Calculate YUV Metrics
        gt_yuv = rgb_to_yuv(gt)
        rec_yuv = rgb_to_yuv(rec)
        
        psnr_y = psnr(rec_yuv[:, 0:1, :, :], gt_yuv[:, 0:1, :, :], data_range=1.0).item()
        psnr_u = psnr(rec_yuv[:, 1:2, :, :], gt_yuv[:, 1:2, :, :], data_range=1.0).item()
        psnr_v = psnr(rec_yuv[:, 2:3, :, :], gt_yuv[:, 2:3, :, :], data_range=1.0).item()

        # Adaptive search for bitstream files
        bits_file = None
        for b_cand in [f"{base_no_ext}.pt", f"bits_{base_no_ext}.pt", f"{base_no_ext}.bin", f"bits_{base_no_ext}.bin",
                      f"{base_no_ext}.h264", f"{base_no_ext}.hevc", f"{base_no_ext}.ivf", f"{base_no_ext}.266"]:
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
        metrics["psnr_y"].append(psnr_y)
        metrics["psnr_u"].append(psnr_u)
        metrics["psnr_v"].append(psnr_v)
        metrics["ssim"].append(ssim_val)
        metrics["lpips"].append(lpips_val)
        metrics["bpp"].append(bpp_val)

        # Save granular data for individual tracking profile
        res_entry = {
            "image_name": r_file,
            "psnr": round(psnr_val, 4),
            "psnr_y": round(psnr_y, 4),
            "psnr_u": round(psnr_u, 4),
            "psnr_v": round(psnr_v, 4),
            "ssim": round(float(ssim_val), 4),
            "lpips": round(lpips_val, 4),
            "bpp": round(bpp_val, 4),
            "qp": qp_map.get(base_no_ext)
        }
        if args.use_vmaf:
            res_entry["vmaf"] = round(vmaf_val, 4)
            
        per_image_results.append(res_entry)


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