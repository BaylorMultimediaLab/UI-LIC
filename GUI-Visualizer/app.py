import streamlit as st
import os
import sys
import subprocess
import glob
from PIL import Image
import json
import importlib.util
import inspect
import tkinter as tk
from tkinter import filedialog

# Add project root to path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

# Import BaseInterface for type checking if needed
from base_interface import BaseInterface

st.set_page_config(page_title="LIC Model Visualizer", layout="wide")

def select_folder(label, key):
    if key not in st.session_state:
        st.session_state[key] = ""
    
    col1, col2 = st.columns([0.8, 0.2])
    with col1:
        path = st.text_input(label, value=st.session_state[key], key=f"input_{key}")
    with col2:
        st.write(" ") # Padding
        if st.button("Browse...", key=f"btn_{key}"):
            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)
            selected_path = filedialog.askdirectory(master=root)
            root.destroy()
            if selected_path:
                st.session_state[key] = selected_path
                st.rerun()
    return st.session_state[key]

def select_file(label, key):
    if key not in st.session_state:
        st.session_state[key] = ""
    
    col1, col2 = st.columns([0.8, 0.2])
    with col1:
        path = st.text_input(label, value=st.session_state[key], key=f"input_{key}")
    with col2:
        st.write(" ") # Padding
        if st.button("Browse...", key=f"btn_{key}"):
            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)
            selected_path = filedialog.askopenfilename(master=root)
            root.destroy()
            if selected_path:
                st.session_state[key] = selected_path
                st.rerun()
    return st.session_state[key]

def load_interfaces(directory):
    registry = {}
    if not os.path.isdir(directory):
        return registry

    for filename in os.listdir(directory):
        if filename.endswith(".py") and not filename.startswith("__"):
            filepath = os.path.join(directory, filename)
            module_name = filename[:-3]

            spec = importlib.util.spec_from_file_location(module_name, filepath)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                try:
                    spec.loader.exec_module(module)
                    for name, obj in inspect.getmembers(module, inspect.isclass):
                        if hasattr(obj, 'TASK_NAME') and getattr(obj, 'TASK_NAME') is not None:
                            registry[obj.TASK_NAME] = obj
                except Exception as e:
                    st.error(f"Failed to load {filename}: {e}")
    return registry

def image_comparison_slider(img1_path, img2_path, label1="Original", label2="Compressed"):
    import base64
    
    def get_image_base64(path):
        with open(path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()

    try:
        img1_b64 = get_image_base64(img1_path)
        img2_b64 = get_image_base64(img2_path)
    except Exception as e:
        st.error(f"Error loading images for comparison: {e}")
        return
    
    html_code = f"""
    <div class="comparison-container" style="position: relative; width: 100%; max-width: 800px; margin: auto; overflow: hidden; user-select: none;">
        <div class="img-container" style="position: relative; width: 100%;">
            <img src="data:image/png;base64,{img1_b64}" style="width: 100%; display: block;">
            <div class="overlay" style="position: absolute; top: 0; left: 0; width: 50%; height: 100%; overflow: hidden; border-right: 2px solid white;">
                <img src="data:image/png;base64,{img2_b64}" style="width: 800px; height: auto; position: absolute; top: 0; left: 0;">
            </div>
            <input type="range" min="0" max="100" value="50" class="slider" style="position: absolute; top: 50%; left: 0; width: 100%; transform: translateY(-50%); opacity: 0; cursor: ew-resize; height: 100%; z-index: 10;">
            <div class="slider-line" style="position: absolute; top: 0; left: 50%; width: 2px; height: 100%; background: white; pointer-events: none; z-index: 5;"></div>
            <div class="slider-button" style="position: absolute; top: 50%; left: 50%; width: 40px; height: 40px; background: white; border: 2px solid #333; border-radius: 50%; transform: translate(-50%, -50%); pointer-events: none; z-index: 6; display: flex; align-items: center; justify-content: center; font-weight: bold;">↔</div>
        </div>
        <div style="display: flex; justify-content: space-between; padding: 10px; font-family: sans-serif;">
            <span>{label1}</span>
            <span>{label2}</span>
        </div>
    </div>
    <script>
        (function() {{
            const container = document.currentScript.previousElementSibling;
            const slider = container.querySelector('.slider');
            const overlay = container.querySelector('.overlay');
            const sliderLine = container.querySelector('.slider-line');
            const sliderButton = container.querySelector('.slider-button');
            const overlayImg = overlay.querySelector('img');

            function updateSlider() {{
                const val = slider.value;
                overlay.style.width = val + '%';
                sliderLine.style.left = val + '%';
                sliderButton.style.left = val + '%';
                
                // Adjust overlay image width to match the base image
                const containerWidth = container.querySelector('.img-container').offsetWidth;
                overlayImg.style.width = containerWidth + 'px';
            }}

            slider.addEventListener('input', updateSlider);
            window.addEventListener('resize', updateSlider);
            // Initial call
            setTimeout(updateSlider, 100);
        }})();
    </script>
    <style>
        .comparison-container .slider:hover ~ .slider-button {{
            background: #eee;
        }}
    </style>
    """
    st.components.v1.html(html_code, height=600)

def main():
    st.title("🖼️ LIC Model Visualization Tool")
    
    test_interfaces_dir = os.path.join(ROOT_DIR, "Interfaces", "Testing-Interfaces")
    registry = load_interfaces(test_interfaces_dir)
    
    st.sidebar.header("1. Data Configuration")
    gt_dir = select_folder("Ground Truth Images Directory", "gt_dir_global")
    
    st.sidebar.header("2. Model Selection")
    selected_models = st.sidebar.multiselect("Select Models to Compare", options=list(registry.keys()))
    
    model_configs = {}
    for model_name in selected_models:
        with st.sidebar.expander(f"Config for {model_name}"):
            interface_cls = registry[model_name]
            
            # Create a dictionary for arguments
            args = {}
            for req in getattr(interface_cls, 'REQUIRED_ARGS', []):
                if req in ['data', 'dataset', 'input', 'input_dir']:
                    continue # We set this globally
                
                if req in ['checkpoint', 'model_path', 'model', 'checkpoints']:
                    args[req] = select_file(f"{req} (Required)", f"{model_name}_{req}")
                else:
                    args[req] = st.text_input(f"{req} (Required)", key=f"{model_name}_{req}")
            
            # Default vars
            default_vars = getattr(interface_cls, 'DEFAULT_VARS', {})
            for key, val in default_vars.items():
                if key in args or key in ['data', 'dataset', 'input', 'input_dir']:
                    continue
                if isinstance(val, bool):
                    args[key] = st.checkbox(key, value=val, key=f"{model_name}_{key}")
                elif key in ['checkpoint', 'model_path', 'model', 'checkpoints']:
                    args[key] = select_file(key, f"{model_name}_{key}")
                else:
                    args[key] = st.text_input(key, value=str(val), key=f"{model_name}_{key}")
            
            working_dir = st.text_input("Working Directory", value=f"LIC-Models/{model_name}", key=f"{model_name}_workdir")
            env_path = select_folder("Env Path (optional)", f"{model_name}_env")
            
            model_configs[model_name] = {
                "args": args,
                "working_dir": working_dir,
                "env_path": env_path
            }
            
    output_base_dir = select_folder("Output Directory", "output_base_dir_global")
    if not output_base_dir:
        output_base_dir = "GUI-Visualizer/outputs"
    
    if st.sidebar.button("🚀 Run Evaluation"):
        if not selected_models:
            st.error("Please select at least one model.")
        elif not gt_dir or not os.path.isdir(gt_dir):
            st.error("Please provide a valid Ground Truth Images Directory.")
        else:
            for model_name in selected_models:
                st.subheader(f"Running {model_name}...")
                config = model_configs[model_name]
                interface_cls = registry[model_name]
                
                # Merge GT dir into args
                final_args = config["args"].copy()
                
                # UNIFIED DATASET INPUT: Map gt_dir to the correct interface key
                # We check ALIASES and REQUIRED_ARGS
                aliases = getattr(interface_cls, 'ALIASES', {})
                rev_aliases = {v: k for k, v in aliases.items()}
                
                # Common data keys in order of preference
                data_keys = ['input', 'input_dir', 'data', 'dataset']
                target_key = None
                
                for k in data_keys:
                    if k in interface_cls.REQUIRED_ARGS or k in interface_cls.DEFAULT_VARS:
                        target_key = k
                        break
                
                if not target_key:
                    # Try aliases
                    for k, v in aliases.items():
                        if v in data_keys:
                            target_key = v
                            break
                
                if not target_key:
                    target_key = "dataset" # Fallback
                
                final_args[target_key] = gt_dir
                
                # Set output dir
                model_output_dir = os.path.join(ROOT_DIR, output_base_dir, model_name)
                os.makedirs(model_output_dir, exist_ok=True)
                
                # Some models take save_dir or output_path
                if model_name == "LIC-HPCM":
                    final_args["save_dir"] = model_output_dir
                elif model_name == "StableCodec":
                    final_args["rec_path"] = model_output_dir
                    final_args["bin_path"] = os.path.join(model_output_dir, "bins")
                elif model_name == "DCVC-RT":
                    final_args["save_dir"] = model_output_dir
                elif model_name == "ELIC":
                    final_args["experiment"] = f"gui_eval_{model_name}"
                elif model_name == "RwkvCompress":
                    final_args["output_dir"] = model_output_dir
                
                # Instantiate interface
                try:
                    interface = interface_cls(job_args=final_args)
                    interface.WORKING_DIR = os.path.join(ROOT_DIR, config["working_dir"])
                    if config["env_path"]:
                        interface.ENV_PATH = config["env_path"]
                    
                    with st.spinner(f"Executing {model_name}..."):
                        interface.execute()
                    st.success(f"Finished {model_name}")
                except Exception as e:
                    st.error(f"Error running {model_name}: {e}")

    # Results Visualization
    if os.path.exists(output_base_dir):
        st.header("🔍 Results Comparison")
        
        gt_images = sorted([f for f in os.listdir(gt_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]) if gt_dir and os.path.isdir(gt_dir) else []
        
        if gt_images:
            selected_img = st.selectbox("Select Image to Compare", gt_images)
            
            cols = st.columns(len(selected_models) if selected_models else 1)
            
            for i, model_name in enumerate(selected_models):
                with cols[i]:
                    st.subheader(f"Model: {model_name}")
                    model_output_dir = os.path.join(output_base_dir, model_name)
                    
                    # Try to find the reconstructed image
                    possible_paths = [
                        os.path.join(model_output_dir, selected_img),
                        os.path.join(model_output_dir, f"rec_{selected_img}"),
                        os.path.join(model_output_dir, selected_img.replace(".jpg", ".png").replace(".jpeg", ".png")),
                    ]
                    
                    # ELIC specific search
                    if model_name == "ELIC":
                        elic_exp_dir = os.path.join(ROOT_DIR, "LIC-Models/ELIC/experiments", f"gui_eval_{model_name}", "codestream")
                        if os.path.exists(elic_exp_dir):
                            epoch_dirs = sorted(os.listdir(elic_exp_dir))
                            if epoch_dirs:
                                last_epoch = epoch_dirs[-1]
                                possible_paths.append(os.path.join(elic_exp_dir, last_epoch, selected_img))
                                possible_paths.append(os.path.join(elic_exp_dir, last_epoch, selected_img.replace(".jpg", ".png").replace(".jpeg", ".png")))

                    # DCVC-RT specific search
                    if model_name == "DCVC-RT":
                        base_name = os.path.splitext(selected_img)[0]
                        if os.path.exists(model_output_dir):
                            for f in os.listdir(model_output_dir):
                                if f.startswith(f"{base_name}_q") and f.endswith(".png"):
                                    possible_paths.append(os.path.join(model_output_dir, f))
                                    break

                    found_img = None
                    for p in possible_paths:
                        if os.path.exists(p):
                            found_img = p
                            break
                    
                    if found_img:
                        gt_img_path = os.path.join(gt_dir, selected_img)
                        image_comparison_slider(gt_img_path, found_img, label1="GT", label2=model_name)
                    else:
                        st.warning(f"No output found for {selected_img} in {model_name}")
        else:
            st.info("Run evaluation or select a valid GT directory to see results.")

if __name__ == "__main__":
    main()
