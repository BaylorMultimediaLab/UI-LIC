#!/usr/bin/env python3
"""
Kodak Dataset Auto-Downloader
Downloads the 24 standard Kodak image dataset files (kodim01.png to kodim24.png)
to the specified target directory (default: ./data/kodak).
"""

import os
import sys
import urllib.request

KODAK_URL_TEMPLATE = "http://r0k.us/graphics/kodak/kodak/kodim{:02d}.png"
FALLBACK_URL_TEMPLATE = "https://raw.githubusercontent.com/kahra/kodak-dataset/master/kodim{:02d}.png"

def download_kodak(target_dir="./data/kodak"):
    target_dir = os.path.abspath(os.path.expanduser(target_dir))
    os.makedirs(target_dir, exist_ok=True)
    
    print(f"Verifying Kodak dataset in: {target_dir}")
    downloaded = 0
    
    for i in range(1, 25):
        filename = f"kodim{i:02d}.png"
        filepath = os.path.join(target_dir, filename)
        
        if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
            continue
            
        print(f"  Downloading {filename}...")
        url = KODAK_URL_TEMPLATE.format(i)
        try:
            urllib.request.urlretrieve(url, filepath)
            downloaded += 1
        except Exception as e:
            print(f"  Primary mirror failed ({e}), trying fallback mirror...")
            fallback_url = FALLBACK_URL_TEMPLATE.format(i)
            try:
                urllib.request.urlretrieve(fallback_url, filepath)
                downloaded += 1
            except Exception as e2:
                print(f"  [ERROR] Failed to download {filename}: {e2}")

    files = [f for f in os.listdir(target_dir) if f.startswith("kodim") and f.endswith(".png")]
    print(f"Kodak Dataset Verification: {len(files)}/24 images present in {target_dir}.")
    return target_dir, len(files) == 24

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "./data/kodak"
    download_kodak(target)
