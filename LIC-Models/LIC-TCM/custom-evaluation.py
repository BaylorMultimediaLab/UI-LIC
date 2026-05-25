import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image  # <-- Added for saving decoded images
from models import TCM
import warnings
import os
import sys
import math
import argparse
import time
from pytorch_msssim import ms_ssim
from PIL import Image

warnings.filterwarnings("ignore")

print(f"CUDA Available: {torch.cuda.is_available()}")

def compute_psnr(a, b):
    mse = torch.mean((a - b)**2).item()
    if mse == 0: return 100.0
    return -10 * math.log10(mse)

def compute_msssim(a, b):
    return -10 * math.log10(1-ms_ssim(a, b, data_range=1.).item())

def compute_bpp(out_net):
    size = out_net['x_hat'].size()
    num_pixels = size[0] * size[2] * size[3]
    return sum(torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
              for likelihoods in out_net['likelihoods'].values()).item()

def pad(x, p):
    h, w = x.size(2), x.size(3)
    new_h = (h + p - 1) // p * p
    new_w = (w + p - 1) // p * p
    padding_left = (new_w - w) // 2
    padding_right = new_w - w - padding_left
    padding_top = (new_h - h) // 2
    padding_bottom = new_h - h - padding_top
    x_padded = F.pad(
        x,
        (padding_left, padding_right, padding_top, padding_bottom),
        mode="constant",
        value=0,
    )
    return x_padded, (padding_left, padding_right, padding_top, padding_bottom)

def crop(x, padding):
    return F.pad(
        x,
        (-padding[0], -padding[1], -padding[2], -padding[3]),
    )

def parse_args(argv):
    parser = argparse.ArgumentParser(description="TCM Testing Script.")
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    parser.add_argument("--data", type=str, help="Path to dataset")
    parser.add_argument("--real", action="store_true", default=True)

    parser.add_argument("-N", type=int, default=128, help="Number of channels")
    parser.add_argument("-M", type=int, default=320, help="Number of channels in bottleneck")
    parser.add_argument("--model", type=str, default="bmshj2018-factorized", help="Model architecture")
    
    # --- ADDED: The save directory argument ---
    parser.add_argument("--save_dir", type=str, default=None, help="Directory to save bitstreams and decoded images")
    
    parser.set_defaults(real=False)
    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv)
    p = 128
    path = args.data
    img_list = []
    
    for file in os.listdir(path):
        if file.lower().endswith((".jpg", ".png", ".jpeg")):
            img_list.append(file)
            
    device = 'cuda:0' if args.cuda else 'cpu'
    
    count = 0
    PSNR = 0
    Bit_rate = 0
    MS_SSIM = 0
    total_time = 0
    dictory = {}
    
    if args.checkpoint:  
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        for k, v in checkpoint["state_dict"].items():
            dictory[k.replace("module.", "")] = v
            
        # Auto-detect N and M from checkpoint to prevent size mismatch
        if 'g_a.0.conv1.weight' in dictory:
            args.N = dictory['g_a.0.conv1.weight'].shape[0] // 2
        if 'g_a.9.weight' in dictory:
            args.M = dictory['g_a.9.weight'].shape[0]
            
    net = TCM(config=[2,2,2,2,2,2], head_dim=[8, 16, 32, 32, 16, 8], drop_path_rate=0.0, N=args.N, M=args.M)
    net = net.to(device)
    net.eval()
    
    if args.checkpoint:
        net.load_state_dict(dictory)

    # --- SETUP OUTPUT DIRECTORIES ---
    if args.save_dir:
        bitstream_dir = os.path.join(args.save_dir, "bitstreams")
        recon_dir = os.path.join(args.save_dir, "reconstruction")
        os.makedirs(bitstream_dir, exist_ok=True)
        os.makedirs(recon_dir, exist_ok=True)
        print(f"Output directories ready at: {args.save_dir}")

    if args.real:
        net.update()
        for img_name in sorted(img_list):
            img_path = os.path.join(path, img_name)
            img = transforms.ToTensor()(Image.open(img_path).convert('RGB')).to(device)
            x = img.unsqueeze(0)
            x_padded, padding = pad(x, p)
            count += 1
            
            with torch.no_grad():
                if args.cuda: torch.cuda.synchronize()
                s = time.time()
                
                out_enc = net.compress(x_padded)
                out_dec = net.decompress(out_enc["strings"], out_enc["shape"])
                
                if args.cuda: torch.cuda.synchronize()
                e = time.time()
                total_time += (e - s)
                
                out_dec["x_hat"] = crop(out_dec["x_hat"], padding)

                # --- SAVE ARTIFACTS ---
                if args.save_dir:
                    base_name = os.path.splitext(img_name)[0]
                    
                    # Save Bitstream (Normalized: kodim01.pt)
                    bit_path = os.path.join(bitstream_dir, f"{base_name}.pt")
                    torch.save({"strings": out_enc["strings"], "shape": out_enc["shape"]}, bit_path)
                    
                    # Save Decoded Image (Normalized: kodim01.png)
                    rec_path = os.path.join(recon_dir, f"{base_name}.png")
                    save_image(out_dec["x_hat"].clamp(0.0, 1.0), rec_path)
                # -----------------------

                num_pixels = x.size(0) * x.size(2) * x.size(3)
                img_bpp = sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels
                img_msssim = compute_msssim(x, out_dec["x_hat"])
                img_psnr = compute_psnr(x, out_dec["x_hat"])
                
                print(f'{img_name} | Bitrate: {img_bpp:.3f}bpp | MS-SSIM: {img_msssim:.2f}dB | PSNR: {img_psnr:.2f}dB')
                
                Bit_rate += img_bpp
                PSNR += img_psnr
                MS_SSIM += img_msssim

    else:
        for img_name in sorted(img_list):
            img_path = os.path.join(path, img_name)
            img = Image.open(img_path).convert('RGB')
            x = transforms.ToTensor()(img).unsqueeze(0).to(device)
            x_padded, padding = pad(x, p)
            count += 1
            
            with torch.no_grad():
                if args.cuda: torch.cuda.synchronize()
                s = time.time()
                
                out_net = net.forward(x_padded)
                
                if args.cuda: torch.cuda.synchronize()
                e = time.time()
                total_time += (e - s)
                
                out_net['x_hat'].clamp_(0, 1)
                out_net["x_hat"] = crop(out_net["x_hat"], padding)
                
                # --- SAVE RECONSTRUCTION (Forward Mode) ---
                if args.save_dir:
                    base_name = os.path.splitext(img_name)[0]
                    # Normalized: kodim01.png
                    rec_path = os.path.join(recon_dir, f"{base_name}.png")
                    save_image(out_net["x_hat"].clamp(0.0, 1.0), rec_path)

                print(f'{img_name} | PSNR: {compute_psnr(x, out_net["x_hat"]):.2f}dB | MS-SSIM: {compute_msssim(x, out_net["x_hat"]):.2f}dB | Bit-rate: {compute_bpp(out_net):.3f}bpp')
                PSNR += compute_psnr(x, out_net["x_hat"])
                MS_SSIM += compute_msssim(x, out_net["x_hat"])
                Bit_rate += compute_bpp(out_net)

    if count > 0:
        PSNR = PSNR / count
        MS_SSIM = MS_SSIM / count
        Bit_rate = Bit_rate / count
        total_time = total_time / count
        print(f'\n--- Average Results ---')
        print(f'average_PSNR: {PSNR:.2f}dB')
        print(f'average_MS-SSIM: {MS_SSIM:.4f}dB')
        print(f'average_Bit-rate: {Bit_rate:.3f} bpp')
        print(f'average_time: {total_time:.3f} ms')
    else:
        print("No images were processed.")

if __name__ == "__main__":
    main(sys.argv[1:])