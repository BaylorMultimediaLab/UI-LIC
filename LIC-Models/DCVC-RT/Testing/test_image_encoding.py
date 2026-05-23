import argparse
import os
import time
import subprocess
import tempfile
import re

import torch
import torchvision.transforms as transforms
from PIL import Image

# Import LPIPS (will gracefully disable if not installed)
try:
    import lpips
    HAS_LPIPS = True
except ImportError:
    HAS_LPIPS = False

from src.models.image_model import DMCI
from src.utils.common import get_state_dict
from src.utils.metrics import calc_psnr, calc_msssim_rgb
from src.layers.cuda_inference import replicate_pad

def parse_args():
    parser = argparse.ArgumentParser(description="Test images with DCVC-RT DMCI (I-frame)")
    parser.add_argument('--model_path', type=str, required=True, help="Path to DMCI model")
    parser.add_argument('--input', type=str, required=True, help="Path to input image or directory")
    parser.add_argument('--qp', type=int, default=27, help="Quantization parameter")
    parser.add_argument('--device', type=str, default="cuda", help="cpu or cuda")
    
    parser.add_argument('--rec_path', type=str, required=True)
    parser.add_argument('--bin_path', type=str, required=True)
    return parser.parse_args()

def calculate_vmaf(orig_path, recon_img):
    """Calculates VMAF using FFmpeg subprocess. Requires FFmpeg with libvmaf."""
    vmaf_score = 0.0
    
    # Save the reconstructed image to a temporary file
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        recon_tmp_path = tmp.name
    
    try:
        recon_img.save(recon_tmp_path)
        
        # Run FFmpeg to calculate VMAF
        cmd = [
            "ffmpeg", "-i", recon_tmp_path, "-i", orig_path,
            "-lavfi", "libvmaf", "-f", "null", "-"
        ]
        
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Parse the VMAF score from FFmpeg's stderr output
        match = re.search(r"VMAF score: ([0-9.]+)", result.stderr)
        if match:
            vmaf_score = float(match.group(1))
        else:
            print(" [WARNING] VMAF failed: Is FFmpeg installed with --enable-libvmaf?")
            
    except Exception as e:
        print(f" [WARNING] VMAF subprocess error: {e}")
    finally:
        if os.path.exists(recon_tmp_path):
            os.remove(recon_tmp_path)
            
    return vmaf_score

def process_image(img_path, net, lpips_fn, args):
    """Processes a single image and returns the metrics."""
    img = Image.open(img_path).convert('RGB')
    x = transforms.ToTensor()(img).unsqueeze(0).to(args.device)
    
    # Get base filename (e.g., 'kodim01')
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    
    if args.device == "cuda":
        x = x.to(torch.float16)

    pic_height, pic_width = x.shape[2], x.shape[3]
    padding_r, padding_b = DMCI.get_padding_size(pic_height, pic_width, 16)
    x_padded = replicate_pad(x, padding_b, padding_r)

    # --- Compress ---
    with torch.no_grad():
        start = time.time()
        encoded = net.compress(x_padded, args.qp)

        # Save Bitstream as normalized name (e.g., kodim01.bin)
        bin_file = os.path.join(args.bin_path, f"{base_name}.bin")
        with open(bin_file, "wb") as f:
            f.write(encoded["bit_stream"])

        sps = {
            'sps_id': 0, 'height': pic_height, 'width': pic_width, 
            'ec_part': 0, 'use_ada_i': 0
        }

        decoded = net.decompress(encoded["bit_stream"], sps, args.qp)        
        end = time.time()

    # --- Unpad and Clamp ---
    recon = decoded['x_hat'][:, :, :pic_height, :pic_width]
    recon_clamped = torch.clamp(recon, 0, 1)
    
    # --- Convert to PIL Image for Saving & VMAF ---
    to_pil = transforms.ToPILImage()
    recon_img = to_pil(recon_clamped.squeeze(0).cpu().float())

    # --- Metrics Calculation ---
    bpp = len(encoded['bit_stream']) * 8 / (pic_height * pic_width)
    inference_time = (end - start) * 1000
    
    rgb_orig = (x.squeeze(0).cpu().float().numpy() * 255).round()
    rgb_rec = (recon_clamped.squeeze(0).cpu().float().numpy() * 255).round()
    
    psnr = calc_psnr(rgb_orig, rgb_rec)
    msssim = calc_msssim_rgb(rgb_orig, rgb_rec)
    
    lpips_val = 0.0
    if lpips_fn is not None:
        with torch.no_grad():
            x_lpips = x.float() * 2.0 - 1.0
            recon_lpips = recon_clamped.float() * 2.0 - 1.0
            lpips_val = lpips_fn(x_lpips, recon_lpips).item()

    vmaf_val = calculate_vmaf(img_path, recon_img)

    print(f"[{base_name}] BPP: {bpp:.4f} | PSNR: {psnr:.2f} | MS-SSIM: {msssim:.4f} | VMAF: {vmaf_val:.2f} | LPIPS: {lpips_val:.4f}")
    
    # --- Save Decoded Image as normalized name (e.g., kodim01.png) ---
    out_path = os.path.join(args.rec_path, f"{base_name}.png")
    recon_img.save(out_path)

    return bpp, psnr, msssim, vmaf_val, lpips_val, inference_time

def main():
    args = parse_args()
    
    if not HAS_LPIPS:
        print("[WARNING] 'lpips' python package not found. LPIPS score will be 0.0.")
        print("          Install it via: pip install lpips")

    os.makedirs(args.bin_path, exist_ok=True)
    os.makedirs(args.rec_path, exist_ok=True)

    # 1. Init Models
    net = DMCI().to(args.device)
    state_dict = get_state_dict(args.model_path)
    net.load_state_dict(state_dict)
    net.eval()
    net.update()
    
    if args.device == "cuda":
        net.half()

    # Init LPIPS Model (AlexNet backend is standard for compression eval)
    lpips_fn = None
    if HAS_LPIPS:
        lpips_fn = lpips.LPIPS(net='alex').to(args.device)
        lpips_fn.eval()

    # 2. Gather files
    image_paths = []
    if os.path.isdir(args.input):
        valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')
        for root, _, files in os.walk(args.input):
            for file in files:
                if file.lower().endswith(valid_exts):
                    image_paths.append(os.path.join(root, file))
    else:
        image_paths = [args.input]

    if not image_paths:
        print(f"[ERROR] No valid images found in: {args.input}")
        return

    print(f"\n--- Starting Evaluation on {len(image_paths)} image(s) (QP: {args.qp}) ---")
    
    # 3. Process all images
    t_bpp, t_psnr, t_msssim, t_vmaf, t_lpips, t_time = 0, 0, 0, 0, 0, 0
    
    for img_path in image_paths:
        bpp, psnr, msssim, vmaf_val, lpips_val, inf_time = process_image(img_path, net, lpips_fn, args)
        
        t_bpp += bpp
        t_psnr += psnr
        t_msssim += msssim
        t_vmaf += vmaf_val
        t_lpips += lpips_val
        t_time += inf_time

    # 4. Print Averages
    count = len(image_paths)
    print(f"\n--- Average Results for {count} image(s) ---")
    print(f"Avg BPP:     {t_bpp / count:.4f}")
    print(f"Avg PSNR:    {t_psnr / count:.2f} dB")
    print(f"Avg MS-SSIM: {t_msssim / count:.4f}")
    print(f"Avg VMAF:    {t_vmaf / count:.2f}")
    print(f"Avg LPIPS:   {t_lpips / count:.4f}")
    print(f"Avg Time:    {t_time / count:.2f} ms")

if __name__ == "__main__":
    main()