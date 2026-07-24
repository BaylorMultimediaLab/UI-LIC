import os
import sys
import argparse
import importlib.util
import subprocess
import webbrowser

# Dynamically import create-env.py because of the dash in filename
def load_create_env():
    script_path = os.path.join(os.path.dirname(__file__), "create-env.py")
    if not os.path.exists(script_path):
        print(f"Error: Could not find 'create-env.py' at {script_path}")
        sys.exit(1)
        
    spec = importlib.util.spec_from_file_location("create_env", script_path)
    create_env_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(create_env_mod)
    return create_env_mod

# --- CONFIGURATION DATA ---

ENV_DOWNLOAD_LINKS = {
    # "ModelName": "URL",
}

WEIGHTS_DATA = {
    "StableCodec": {
        "base_path": "LIC-Models/StableCodec/weights/",
        "description": "Finetuned for extreme low bitrates (0.005 - 0.035 bpp). Higher 'ft' number = lower bitrate / higher compression.",
        "options": [
            {"name": "stablecodec_base.pkl", "id": "1M8HUsL27sscgFb-DsJDr9QPmW4dC2a-4", "desc": "Base model for finetuning"},
            {"name": "stablecodec_ft2.pkl", "id": "1-rze9I-e4iE9hMDRH152aQLwz9nTUvgE", "desc": "~0.035 bpp (Highest quality)"},
            {"name": "stablecodec_ft3.pkl", "id": "1jrjpsTNv1cb1miZC1zozHgGm-EN3IYHV", "desc": "~0.029 bpp"},
            {"name": "stablecodec_ft4.pkl", "id": "1sj4aFQwMVjce4t6A77YXRiHQlXzOoiUp", "desc": "~0.025 bpp"},
            {"name": "stablecodec_ft6.pkl", "id": "1PiBBCri6pn-JVWiDO0xGb6Olap0N2lga", "desc": "~0.020 bpp"},
            {"name": "stablecodec_ft8.pkl", "id": "1CVXLaVv48vM1Iy_Zqe3wcbJq_Lnh0vNw", "desc": "~0.017 bpp"},
            {"name": "stablecodec_ft12.pkl", "id": "1fp7aut-EAIRk1Ef0cR_R30flqNn5hI9T", "desc": "~0.013 bpp"},
            {"name": "stablecodec_ft16.pkl", "id": "1Yu2F-9BKd9gDq6c9J7cBsC4_gy2bIHg_", "desc": "~0.010 bpp"},
            {"name": "stablecodec_ft24.pkl", "id": "1grLxmLuth4ydXhghZwa9wxGhjnRGzXim", "desc": "~0.008 bpp"},
            {"name": "stablecodec_ft32.pkl", "id": "1quyX_-g4B05DQrMb5bGlonaFrOlLyd8i", "desc": "~0.005 bpp (Highest compression)"},
            {"name": "elic_official.pth", "id": "1jUfYJdZd0-bYUsoOUWwEpI5t1MZYP3AP", "dest_name": "elic.pth", "desc": "Official ELIC auxiliary model"},
        ]
    },
    "ELIC": {
        "base_path": "LIC-Models/ELIC/weights/",
        "description": "Official ELIC pretrained model.",
        "options": [
            {"name": "ELIC_0450_ft_3980_Plateau.pth.tar", "id": "1uuKQJiozcBfgGMJ8CfM6lrXOZWv6RUDN",  "desc": "lambda 0.45 (Highest quality)"},
            {"name": "ELIC_0150_ft_3980_Plateau.pth.tar", "id": "1s544Uxv0gBY3WvKBcGNb3Fb22zfmd9PL",  "desc": "lambda 0.15"},
            {"name": "ELIC_0032_ft_3980_Plateau.pth.tar", "id": "1Moody9IR8CuAGwLCZ_ZMTfZXT0ehQhqc",  "desc": "lambda 0.032"},
            {"name": "ELIC_0016_ft_3980_Plateau.pth.tar", "id": "1MWlYAmpHbWlGtG7MBBTPEew800grY5yC",  "desc": "lambda 0.016"},
            {"name": "ELIC_0008_ft_3980_Plateau.pth.tar", "id": "1VNE7rx-rBFLnNFkz56Zc-cPr6xrBBJdL",  "desc": "lambda 0.008"},
            {"name": "ELIC_0004_ft_3980_Plateau.pth.tar", "id": "1YGVJ9bpeEq0xfqka2xkaMzhDkeYFJi6q",  "desc": "lambda 0.004 (Highest compression)"},
        ]
    },
    "RwkvCompress": {
        "base_path": "LIC-Models/RwkvCompress/weights/",
        "description": "LALIC quality levels based on MSE optimization. Q6 is highest quality / highest bitrate.",
        "options": [
            {"name": "lalic-q1.pth", "id": "1908uXi4ofAUdznLvA-NAKe2rTLJlY2pA", "desc": "Lambda=0.0018 (Highest compression)"},
            {"name": "lalic-q2.pth", "id": "1WckWVqow2GDnXuY7Z4aPHm4ovKUIenh0", "desc": "Lambda=0.0035"},
            {"name": "lalic-q3.pth", "id": "1quDdHXsJPgdgGwTZsJRf0guCru43G17F", "desc": "Lambda=0.0067"},
            {"name": "lalic-q4.pth", "id": "1DtJigNUa80mPYBtjhCOdyPZQ5eRkUf2i", "desc": "Lambda=0.0130"},
            {"name": "lalic-q5.pth", "id": "1W5fwRrPI9KDWwdQ3QszL2rlGkqHAFnAB", "desc": "Lambda=0.0250"},
            {"name": "lalic-q6.pth", "id": "1c2yYxB8Riq5BUrJlZr3bW5mt-y1e8mTe", "desc": "Lambda=0.0483 (Highest quality)"},
        ]
    },
    "HPCM": {
        "base_path": "LIC-Models/HPCM/weights/",
        "description": "Hierarchical Progressive Context Modeling. Base (smaller) and Large variants. Optimized for MSE (Standard) or MS-SSIM (Perceptual).",
        "options": [
            # HPCM-Base MSE
            {"name": "hpcm_base_mse_0.0018.pth", "id": "1nIoANbXzBNE0S_VoLo9ZDHU50lPMmeBP", "desc": "Base, MSE, λ=0.0018 (Low quality)"},
            {"name": "hpcm_base_mse_0.0035.pth", "id": "15J_nl33_5R_qyTIzLAaT60ICn9BMGHlB", "desc": "Base, MSE, λ=0.0035"},
            {"name": "hpcm_base_mse_0.0067.pth", "id": "1HIzsEqAPztaMh0Frqec4TtRwoc7uxO97", "desc": "Base, MSE, λ=0.0067"},
            {"name": "hpcm_base_mse_0.013.pth", "id": "1Snq7vkWQdApzCe-gK_V-WuRyMHQRL443", "desc": "Base, MSE, λ=0.013"},
            {"name": "hpcm_base_mse_0.025.pth", "id": "1NFZD87BkfU28YnDqpzfphG0xDZDZpUA5", "desc": "Base, MSE, λ=0.025"},
            {"name": "hpcm_base_mse_0.0483.pth", "id": "1G5wm4KENBY2qSAQBxNw3Rz4JcMxH8HXu", "desc": "Base, MSE, λ=0.0483 (High quality)"},
            # HPCM-Base MS-SSIM
            {"name": "hpcm_base_ssim_2.4.pth", "id": "1AZ9dY2J9Rn17YSQe_NYIOID-st1C-68O", "desc": "Base, MS-SSIM, λ=2.4"},
            {"name": "hpcm_base_ssim_4.58.pth", "id": "1Y8gEL4MRNB-TBbOMDUKeMTO_z1QhbwqL", "desc": "Base, MS-SSIM, λ=4.58"},
            {"name": "hpcm_base_ssim_8.73.pth", "id": "1hXK-X6GsjjiULy6FvU80Smob_2UOFeFJ", "desc": "Base, MS-SSIM, λ=8.73"},
            {"name": "hpcm_base_ssim_16.64.pth", "id": "1antXt3M0ecOVejbpxL1U7CVx4TS_XPMQ", "desc": "Base, MS-SSIM, λ=16.64"},
            {"name": "hpcm_base_ssim_31.73.pth", "id": "1X_Q0hHwAW0GOsHWLoq84YKYqXrduFe6b", "desc": "Base, MS-SSIM, λ=31.73"},
            {"name": "hpcm_base_ssim_60.5.pth", "id": "1mX885h4eVwLvpeHpBHBoM1p4Z2VLV2y-", "desc": "Base, MS-SSIM, λ=60.5"},
            # HPCM-Large MSE
            {"name": "hpcm_large_mse_0.0018.pth", "id": "1E1DUaPsIrfNPwfk4qD-630hhxx5n_BJ4", "desc": "Large, MSE, λ=0.0018"},
            {"name": "hpcm_large_mse_0.0035.pth", "id": "15yDUVvEBn-7dMA9SBIQ2w28LJXBGntQo", "desc": "Large, MSE, λ=0.0035"},
            {"name": "hpcm_large_mse_0.0067.pth", "id": "1yzZKji6RpsyQPD6KFr_weavVrlmn-V4R", "desc": "Large, MSE, λ=0.0067"},
            {"name": "hpcm_large_mse_0.013.pth", "id": "1L19zjwOpbbFPw0FxnyVLcHATxCaorjUV", "desc": "Large, MSE, λ=0.013"},
            {"name": "hpcm_large_mse_0.025.pth", "id": "1oh8OwCLc8PEVMW1fc9LoC7G4385kHU5D", "desc": "Large, MSE, λ=0.025"},
            {"name": "hpcm_large_mse_0.0483.pth", "id": "1VWLPQeDzBZgb1D2mZ9jLzLppXL8gUanH", "desc": "Large, MSE, λ=0.0483"},
            # HPCM-Large MS-SSIM
            {"name": "hpcm_large_ssim_2.4.pth", "id": "1RUM2a1wdI8Yj9-tvzO_MnHGZWZRp2-W6", "desc": "Large, MS-SSIM, λ=2.4"},
            {"name": "hpcm_large_ssim_4.58.pth", "id": "1TL_QDlfzHvmerN1p0rn5mJbSNwn3LXXx", "desc": "Large, MS-SSIM, λ=4.58"},
            {"name": "hpcm_large_ssim_8.73.pth", "id": "1nIEJY9ecr9uA9XidtiQRXQ2rzm1DWKM0", "desc": "Large, MS-SSIM, λ=8.73"},
            {"name": "hpcm_large_ssim_16.64.pth", "id": "1sKnWry4LIZPawwv08TH3l_41giUuElCx", "desc": "Large, MS-SSIM, λ=16.64"},
            {"name": "hpcm_large_ssim_31.73.pth", "id": "1rR0vFbQ2fOT7EgJbYg5f0OdiIT5jbPPu", "desc": "Large, MS-SSIM, λ=31.73"},
            {"name": "hpcm_large_ssim_60.5.pth", "id": "1ITR5JEzLjmdHLp20GYzIdwE8eEK2d7ns", "desc": "Large, MS-SSIM, λ=60.5"},
        ]
    },
    "LIC-TCM": {
        "base_path": "LIC-Models/LIC-TCM/weights/",
        "description": "Mixed Transformer-CNN architectures. N=128 (Large) or N=64 (Small). Optimized for MSE.",
        "options": [
            {"name": "tcm_mse_128_0.05.pth", "id": "1TK-CPiD2QwtWJqZoT_OyCtnxdQ7UNP56", "desc": "N=128, λ=0.05 (Highest quality)"},
            {"name": "tcm_mse_64_0.05.pth", "id": "1Quz6_jGJyaG6LMUbT4JuOhhQWxJN26Kh", "desc": "N=64, λ=0.05"},
            {"name": "tcm_mse_64_0.025.pth", "id": "1rc4E2Rke1Jd8UnLq73NaXbfAdcBGGPKg", "desc": "N=64, λ=0.025"},
            {"name": "tcm_mse_64_0.013.pth", "id": "1UbfQFsrr-Z6SrvZvpX4p1QPta5FCORZ5", "desc": "N=64, λ=0.013"},
            {"name": "tcm_mse_64_0.0067.pth", "id": "17THA1IiPStSO6jG4h5clwkw0ySzgLZID", "desc": "N=64, λ=0.0067"},
            {"name": "tcm_mse_64_0.0035.pth", "id": "1x2rfIQAv8RsjM3zEByDdOZJtEcPU5XZT", "desc": "N=64, λ=0.0035"},
            {"name": "tcm_mse_64_0.0025.pth", "id": "1zpkW_MCkUWl8nRUlza0L7Fk7dXlXciZd", "desc": "N=64, λ=0.0025 (Highest compression)"},
        ]
    },
    "DCVC-RT": {
        "base_path": "LIC-Models/DCVC-RT/weights/",
        "description": "Deep Context Video Compression (Real-Time).",
        "options": [
            {"name": "cvpr2025_image.pth.tar", "url": "https://1drv.ms/f/c/2866592d5c55df8c/Esu0KJ-I2kxCjEP565ARx_YB88i0UnR6XnODqFcvZs4LcA?e=by8CO8", "desc": "Pretrained Image Model (Manual Download)"},
        ]
    }
}

# --- HELPER FUNCTIONS ---

def download_file(option, base_dir):
    name = option.get("dest_name", option["name"])
    dest_path = os.path.join(base_dir, name)
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        print(f" [SKIP] {name} already exists.")
        return dest_path

    print(f" [DOWNLOADING] {name}...")
    try:
        if "id" in option:
            file_id = option["id"]
            res = subprocess.run([sys.executable, "-m", "gdown", file_id, "-O", dest_path, "--quiet"])
            if res.returncode != 0 or not os.path.exists(dest_path) or os.path.getsize(dest_path) == 0:
                print(f" [INFO] gdown rate-limited. Trying direct curl fallback for {name}...")
                curl_url = f"https://drive.usercontent.google.com/download?id={file_id}&confirm=t"
                subprocess.run(["curl", "-L", "-s", curl_url, "-o", dest_path])
                
                if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
                    sz_mb = os.path.getsize(dest_path) / (1024 * 1024)
                    print(f" [SUCCESS] Downloaded {name} ({sz_mb:.1f} MB) via direct link.")
                else:
                    print(f" [WARNING] Download rate-limited for {name}.")
                    print(f" -> Direct Browser Link: https://drive.google.com/uc?id={file_id}")
                    if os.path.exists(dest_path):
                        os.remove(dest_path)
                    return None
            else:
                sz_mb = os.path.getsize(dest_path) / (1024 * 1024)
                print(f" [SUCCESS] Downloaded {name} ({sz_mb:.1f} MB).")
        elif "url" in option and option["url"] != "TODO":
            subprocess.run(["curl", "-L", option["url"], "-o", dest_path], check=True)
        else:
            print(f" [ERROR] No valid URL or Google Drive ID for {name}")
            return None
        return dest_path
    except Exception as e:
        print(f" [ERROR] Failed to download {name}: {e}")
        return None

def clear_screen():
    sys.stdout.write("\033[H\033[2J")
    sys.stdout.flush()

def handle_manual_downloads(model_name, options, base_dir):
    if not options:
        return
    
    url = options[0].get("url")
    os.makedirs(base_dir, exist_ok=True)
    
    while True:
        clear_screen()
        print(f"\n--- MANUAL DOWNLOAD: {model_name} ---")
        print(f"\n1. URL: {url}")
        print(f"2. PATH: {base_dir}")
        print(f"\n3. STATUS:")
        
        missing_count = 0
        for opt in options:
            exists = os.path.exists(os.path.join(base_dir, opt['name']))
            status = "[FOUND]" if exists else "[MISSING]"
            if not exists: missing_count += 1
            print(f"   {status} {opt['name']}")
        
        if missing_count == 0:
            print(f"\n[SUCCESS] All {model_name} files found.")
            input("Press ENTER to continue...")
            return

        print(f"\n[Commands] [ENTER] Verify | [O]pen Link | [S]kip | [Q]uit")
        resp = input("Choice: ").lower().strip()
        
        if resp == 'o':
            try:
                webbrowser.open(url)
            except Exception:
                print("Failed to open browser. Please copy the URL manually.")
                input("Press ENTER...")
        elif resp == 's':
            return
        elif resp == 'q':
            sys.exit(0)

def select_from_list(items, title, multi=True, descriptions=None):
    print(f"\n--- {title} ---")
    for i, item in enumerate(items):
        desc = f" - {descriptions[i]}" if descriptions and descriptions[i] else ""
        print(f"{i+1}. {item}{desc}")
    
    if multi:
        print("Select multiple (e.g. 1,2,4) or 'all' or 'none': ", end="")
    else:
        print("Select one: ", end="")
        
    choice = input().strip().lower()
    if choice == 'none' or not choice:
        return []
    if choice == 'all' and multi:
        return items
    
    try:
        indices = [int(x.strip()) - 1 for x in choice.split(",")]
        selected = [items[i] for i in indices if 0 <= i < len(items)]
        return selected if multi else (selected[0] if selected else None)
    except:
        print("Invalid selection.")
        return []

# --- MAIN ---

def main():
    parser = argparse.ArgumentParser(description="Unified LIC Quick-Start Script")
    parser.add_argument("base_path", nargs="?", help="Base directory for environments.")
    parser.add_argument("--all", action="store_true", help="Setup everything with defaults.")
    args = parser.parse_args()

    # Path setup
    base_path = args.base_path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "LIC-Models")
    base_path = os.path.abspath(os.path.expanduser(base_path))

    while True:
        # 1. Environment Selection
        models = [
            {"name": "DCVC-RT", "python": "3.12", "req": "LIC-Models/DCVC-RT/requirements.txt"},
            {"name": "ELIC", "python": "3.10", "req": "LIC-Models/ELIC/requirements.txt"},
            {"name": "HPCM", "python": "3.10", "req": "LIC-Models/HPCM/requirements.txt"},
            {"name": "LIC-TCM", "python": "3.10", "req": "LIC-Models/LIC-TCM/requirements.txt"},
            {"name": "RwkvCompress", "python": "3.10", "req": "LIC-Models/RwkvCompress/requirements.txt"},
            {"name": "StableCodec", "python": "3.10", "req": "LIC-Models/StableCodec/requirements.txt"},
            {"name": "eval", "python": "3.10", "req": "evaluation-requirements.txt"}
        ]

        def is_env_setup(m):
            env_name = f"{m['name']}-env"
            env_path = os.path.join(base_path, env_name)
            py_bin = os.path.join(env_path, "bin", "python3")
            py_exe = os.path.join(env_path, "Scripts", "python.exe")
            return os.path.exists(py_bin) or os.path.exists(py_exe) or os.path.exists(env_path)

        model_names = [m["name"] for m in models]
        model_descs = ["(Already setup)" if is_env_setup(m) else "(Not setup)" for m in models]
        selected_models = model_names if args.all else select_from_list(model_names, "Select Models to Setup Environments", descriptions=model_descs)

        # 2. Weights Selection
        weight_models = [m for m in model_names if m in WEIGHTS_DATA]
        
        # Helper to check if model has any missing weights
        def has_missing_weights(m_name):
            data = WEIGHTS_DATA[m_name]
            b_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), data["base_path"])
            for o in data["options"]:
                name = o.get("dest_name", o["name"])
                t_path = os.path.join(b_dir, name)
                if not (os.path.exists(t_path) and os.path.getsize(t_path) > 0):
                    return True
            return False

        weight_descriptions = []
        for m in weight_models:
            status = " (Missing weights)" if has_missing_weights(m) else " (All downloaded)"
            weight_descriptions.append(WEIGHTS_DATA[m]["description"] + status)

        selected_weights = weight_models if args.all else select_from_list(weight_models, "Select Models to Download Weights", descriptions=weight_descriptions)

        # 4. Specific Weight Selection (Detailed)
        weights_to_download = {} # {model_name: [options]}
        if selected_weights:
            for m_name in selected_weights:
                data = WEIGHTS_DATA[m_name]
                b_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), data["base_path"])
                
                # Filter to show ONLY missing weights
                missing_options = []
                for o in data["options"]:
                    name = o.get("dest_name", o["name"])
                    t_path = os.path.join(b_dir, name)
                    if not (os.path.exists(t_path) and os.path.getsize(t_path) > 0):
                        missing_options.append(o)

                if not missing_options:
                    print(f"\n[MODEL INFO] {m_name}: All pretrained weights are ALREADY downloaded!")
                    continue

                if args.all:
                    weights_to_download[m_name] = missing_options
                else:
                    print(f"\n[MODEL INFO] {m_name}: {data['description']}")
                    item_names = [o["name"] for o in missing_options]
                    item_descs = [o.get("desc", "") for o in missing_options]
                    selected_opts = select_from_list(item_names, f"Select missing weights for {m_name}", descriptions=item_descs)
                    if selected_opts:
                        weights_to_download[m_name] = [o for o in missing_options if o["name"] in selected_opts]

        # 5. Final Confirmation
        print("\n" + "="*60)
        print("FINAL SETUP SUMMARY")
        print("="*60)
        print(f"Base Path: {base_path}")
        
        print(f"Environments to Setup:")
        if not selected_models:
            print("  None")
        else:
            for m_name in selected_models:
                env_path = os.path.join(base_path, f"{m_name}-env")
                exists = os.path.exists(env_path)
                status = " (previously installed)" if exists else ""
                print(f"  - {m_name}{status}")

        print(f"Weights to Download:")
        if not weights_to_download:
            print("  None")
        else:
            for m_name, opts in weights_to_download.items():
                print(f"  - {m_name}: {len(opts)} file(s)")
                data = WEIGHTS_DATA[m_name]
                base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), data["base_path"])
                for o in opts:
                    name = o.get("dest_name", o["name"])
                    exists = os.path.exists(os.path.join(base_dir, name))
                    status = " (already exists)" if exists else ""
                    print(f"    * {name}{status}")
        
        if args.all:
            break

        confirm = input("\nProceed with the above plan? [Y]es / [M]odify / [Q]uit: ").strip().lower()
        if confirm == 'y':
            break
        elif confirm == 'q':
            print("Setup cancelled.")
            return
        # If 'm', it loops back for re-selection

    create_env_mod = load_create_env()

    # --- EXECUTION ---

    # Setup Environments
    if selected_models:
        print(f"\n>>> Setting up environments...")
        for m_name in selected_models:
            model = next(m for m in models if m["name"] == m_name)
            env_path = os.path.join(base_path, f"{m_name}-env")
            req_path = os.path.join(os.path.dirname(__file__), model['req'])
            
            if not os.path.exists(req_path):
                print(f" [SKIP] {m_name}: requirements not found.")
                continue

            try:
                if os.path.exists(env_path):
                    print(f" [INFO] {m_name}: Updating existing env.")
                    create_env_mod.update_pip_requirements(env_path, req_path)
                else:
                    print(f" [INFO] {m_name}: Creating new env.")
                    create_env_mod.setup_conda_env(env_path, req_path, model['python'])
            except Exception as e:
                print(f" [ERROR] {m_name} env setup failed: {e}")

    # Download Weights
    if weights_to_download:
        print("\n>>> Downloading Pretrained Weights")
        for m_name, options in weights_to_download.items():
            data = WEIGHTS_DATA[m_name]
            
            # Special case for StableCodec sd-turbo (always check if StableCodec weights are requested)
            if m_name == "StableCodec":
                sd_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "LIC-Models", "StableCodec", "sd-turbo")
                if not os.path.exists(sd_path):
                    print(" [STABLECODEC] Downloading sd-turbo...")
                    try:
                        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "huggingface-hub"], check=True)
                        cmd = f'from huggingface_hub import snapshot_download; snapshot_download(repo_id="stabilityai/sd-turbo", local_dir="{sd_path}")'
                        subprocess.run([sys.executable, "-c", cmd], check=True)
                    except: print("  [ERROR] sd-turbo download failed.")

            base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), data["base_path"])
            if m_name == "DCVC-RT":
                handle_manual_downloads(m_name, options, base_dir)
            else:
                for option in options:
                    dest_path = download_file(option, base_dir)


    print("\nSetup finished.")

if __name__ == "__main__":
    main()
