import os
import subprocess

# --- CONFIGURATION ---
KODAK_DIR = "/mnt/c/Users/Nicholas_Nolen1/data/kodak"
LIST_FILE = "/mnt/c/Users/Nicholas_Nolen1/data/kodak_input.txt"
OUT_DIR = "/mnt/c/Users/Nicholas_Nolen1/Desktop/JPEG-AI/output_dir/KODAK_MANUAL_SUCCESS"
CHECKPOINT = "/home/nicholas/data/jpeg_ai_data/output_dir/beta_0.075/best.pth"
# 75 corresponds to your beta 0.075 model
TARGET_BPP = "75" 

# Ensure output directories exist
os.makedirs(os.path.join(OUT_DIR, "bit"), exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "rec"), exist_ok=True)

# Read image list
with open(LIST_FILE, "r") as f:
    images = [line.strip() for line in f if line.strip()]

print(f"Found {len(images)} images. Starting 4090 Encoding...")

for img_name in images:
    input_path = os.path.join(KODAK_DIR, img_name)
    bit_path = os.path.join(OUT_DIR, "bit", img_name.replace(".png", ".bits"))
    rec_path = os.path.join(OUT_DIR, "rec", img_name)
    
    # We call the core encoder directly to bypass the 'eval' wrapper bugs
    cmd = [
        "python3", "-m", "src.reco.coders.encoder",
        input_path,
        bit_path,
        "-r", rec_path,
        "--checkpoint", CHECKPOINT,
        "--target_bpps", TARGET_BPP,
        "--calc_metrics",
        "--gpu_id", "0"
    ]
    
    print(f"\n>>> Processing: {img_name}")
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print(f"!!! Error processing {img_name}")

print("\nBatch Complete. Check your output_dir for rec/ and bit/ files.")
