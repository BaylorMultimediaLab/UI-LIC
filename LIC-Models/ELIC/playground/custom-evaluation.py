import torch
import torch.nn.functional as F
import os
import time
import logging
import math
import sys
from torchvision import transforms
from torchvision.utils import save_image
from PIL import ImageFile, Image

# Import existing configs
from config.args import test_options
from config.config import model_config
from models import ELIC
from utils.logger import setup_logger

import faulthandler
faulthandler.enable()

def pad(x, p=64):
    h, w = x.size(2), x.size(3)
    H = (h + p - 1) // p * p
    W = (w + p - 1) // p * p
    padding_left = (W - w) // 2
    padding_right = W - w - padding_left
    padding_top = (H - h) // 2
    padding_bottom = H - h - padding_top
    return F.pad(x, (padding_left, padding_right, padding_top, padding_bottom), mode="constant", value=0), (padding_left, padding_right, padding_top, padding_bottom)

def crop(x, padding):
    return F.pad(x, (-padding[0], -padding[1], -padding[2], -padding[3]))

def main():
    torch.backends.cudnn.deterministic = True
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    Image.MAX_IMAGE_PIXELS = None

    # --- 1. EXTRACT PATHS FROM INTERFACE ---
    rec_path, bin_path = None, None
    
    if "--rec_path" in sys.argv:
        idx = sys.argv.index("--rec_path")
        rec_path = sys.argv[idx + 1]
        sys.argv.pop(idx)
        sys.argv.pop(idx)
        
    if "--bin_path" in sys.argv:
        idx = sys.argv.index("--bin_path")
        bin_path = sys.argv[idx + 1]
        sys.argv.pop(idx)
        sys.argv.pop(idx)

    # --- 2. PARSE STANDARD ELIC ARGS ---
    args = test_options()
    config = model_config()

    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    if not os.path.exists(os.path.join('./experiments', args.experiment)):
        os.makedirs(os.path.join('./experiments', args.experiment))
    setup_logger('test', os.path.join('./experiments', args.experiment), 'test_' + args.experiment, level=logging.INFO, screen=True, tofile=True)
    logger_test = logging.getLogger('test')

    net = ELIC(config=config).to(device)
    
    logger_test.info(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    net.load_state_dict(checkpoint['state_dict'])
    net.update(force=True)
    net.eval()
    
    logger_test.info("Start testing!")

    # --- 3. MANUAL EVALUATION LOOP ---
    # We use a manual loop instead of DataLoader so we can keep the exact filenames for saving
    img_list = [f for f in os.listdir(args.dataset) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    transform = transforms.ToTensor()

    for img_name in sorted(img_list):
        img_path = os.path.join(args.dataset, img_name)
        img = transform(Image.open(img_path).convert('RGB')).unsqueeze(0).to(device)
        
        x_padded, padding = pad(img, p=64) 
        
        with torch.no_grad():
            if args.cuda: torch.cuda.synchronize()
            
            # Compress and Decompress
            out_enc = net.compress(x_padded)
            out_dec = net.decompress(out_enc["strings"], out_enc["shape"])
            
            if args.cuda: torch.cuda.synchronize()
            
            x_hat = crop(out_dec["x_hat"], padding)
            
            num_pixels = img.size(2) * img.size(3)
            img_bpp = sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels

            # --- 4. SAVE ARTIFACTS USING THE EXTRACTED PATHS ---
            if bin_path and rec_path:
                base_name = os.path.splitext(img_name)[0]
                
                # Save Bitstream
                bit_file = os.path.join(bin_path, f"{base_name}.pt")
                torch.save({"strings": out_enc["strings"], "shape": out_enc["shape"]}, bit_file)
                
                # Save Reconstruction
                rec_file = os.path.join(rec_path, f"{img_name}")
                save_image(x_hat.clamp(0.0, 1.0), rec_file)
            # ---------------------------------------------------

            logger_test.info(f"{img_name} | BPP: {img_bpp:.3f}")

if __name__ == '__main__':
    main()