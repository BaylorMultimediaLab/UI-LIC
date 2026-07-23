import sys
import os
import argparse
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torchvision.transforms as transforms
from PIL import Image

from src.models.image_model import DMCI
from src.utils.common import get_state_dict
from src.utils.transforms import rgb2ycbcr, ycbcr2rgb
from src.layers.cuda_inference import replicate_pad


def parse_args():
    parser = argparse.ArgumentParser(description="Test images with DCVC-RT DMCI (I-frame)")
    parser.add_argument('--model_path', type=str, required=True, help="Path to DMCI model")
    parser.add_argument('--input',      type=str, required=True, help="Path to input image or directory")
    parser.add_argument('--qp',         type=int, default=27,    help="Quantization parameter")
    parser.add_argument('--device',     type=str, default="cuda", help="cpu or cuda")
    parser.add_argument('--rec_path',   type=str, required=True, help="Directory for reconstructed PNGs")
    parser.add_argument('--bin_path',   type=str, required=True, help="Directory for bitstreams")
    parser.add_argument('--metrics',    action='store_true',
                        help="Compute inline PSNR/MS-SSIM for debugging. "
                             "All metrics are recomputed by evaluation.py anyway.")
    return parser.parse_args()


def process_image(img_path, net, args):
    """Compress one image, save bitstream and reconstruction. Returns bpp and inference time."""
    img = Image.open(img_path).convert('RGB')
    x_rgb = transforms.ToTensor()(img).unsqueeze(0).to(args.device)

    base_name = os.path.splitext(os.path.basename(img_path))[0]

    if args.device.startswith("cuda"):
        x_rgb = x_rgb.to(torch.float16)

    # DCVC-RT expects YCbCr
    x = rgb2ycbcr(x_rgb)
    pic_height, pic_width = x.shape[2], x.shape[3]
    padding_r, padding_b = DMCI.get_padding_size(pic_height, pic_width, 16)
    x_padded = replicate_pad(x, padding_b, padding_r)

    # --- Compress + Decompress ---
    with torch.no_grad():
        start = time.time()
        use_two_entropy_coders = pic_height * pic_width > 1280 * 720
        net.set_use_two_entropy_coders(use_two_entropy_coders)

        encoded = net.compress(x_padded, args.qp)

        bin_file = os.path.join(args.bin_path, f"{base_name}.bin")
        with open(bin_file, "wb") as f:
            f.write(encoded["bit_stream"])

        sps = {
            'sps_id': 0, 'height': pic_height, 'width': pic_width,
            'ec_part': 1 if use_two_entropy_coders else 0, 'use_ada_i': 0
        }
        decoded = net.decompress(encoded["bit_stream"], sps, args.qp)
        end = time.time()

    # --- Unpad, clamp, convert back to RGB ---
    recon = decoded['x_hat'][:, :, :pic_height, :pic_width]
    recon_rgb = ycbcr2rgb(torch.clamp(recon, 0, 1))
    recon_img = transforms.ToPILImage()(recon_rgb.squeeze(0).cpu().float())

    # --- Save reconstruction ---
    recon_img.save(os.path.join(args.rec_path, f"{base_name}.png"))

    bpp = len(encoded['bit_stream']) * 8 / (pic_height * pic_width)
    inference_time = (end - start) * 1000

    # Optional inline metrics (debug only - evaluation.py is authoritative)
    if args.metrics:
        from src.utils.metrics import calc_psnr, calc_msssim_rgb
        rgb_orig = (x_rgb.squeeze(0).cpu().float().numpy() * 255).round()
        rgb_rec  = (recon_rgb.squeeze(0).cpu().float().numpy() * 255).round()
        psnr   = calc_psnr(rgb_orig, rgb_rec)
        msssim = calc_msssim_rgb(rgb_orig, rgb_rec)
        print(f"[{base_name}] bpp: {bpp:.4f} | PSNR: {psnr:.2f} | MS-SSIM: {msssim:.4f} | {inference_time:.1f}ms")
    else:
        print(f"[{base_name}] bpp: {bpp:.4f} | {inference_time:.1f}ms")

    return bpp, inference_time


def main():
    args = parse_args()

    os.makedirs(args.bin_path, exist_ok=True)
    os.makedirs(args.rec_path, exist_ok=True)

    # Load model
    net = DMCI().to(args.device)
    net.load_state_dict(get_state_dict(args.model_path))
    net.eval()
    net.update()

    if args.device.startswith("cuda"):
        net.half()

    # Gather input images
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

    print(f"\n--- Encoding {len(image_paths)} image(s) at QP={args.qp} ---")
    print("    (Full metrics computed by evaluation.py from saved reconstructions)")

    t_bpp, t_time = 0.0, 0.0
    for img_path in image_paths:
        bpp, inf_time = process_image(img_path, net, args)
        t_bpp  += bpp
        t_time += inf_time

    count = len(image_paths)
    print(f"\n--- Summary ({count} images, QP={args.qp}) ---")
    print(f"Avg bpp:  {t_bpp  / count:.4f}")
    print(f"Avg time: {t_time / count:.2f} ms")
    print("[SUCCESS] Script completed.")


if __name__ == "__main__":
    main()
