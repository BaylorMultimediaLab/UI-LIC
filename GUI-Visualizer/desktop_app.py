import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import sys

# Add parent directory to sys.path to allow importing from root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import threading
import importlib.util
import inspect
from PIL import Image, ImageTk
import glob
import builtins
import queue
import json
import subprocess
import time
from dispatcher import Dispatcher

try:
    from ctypes import windll
    windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    pass

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

class ComparisonCanvas(tk.Canvas):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.image1 = None # Right Image
        self.image2 = None # Left Image
        self.label1 = "Right"
        self.label2 = "Left"
        self.metrics1 = "" # Right metrics string
        self.metrics2 = "" # Left metrics string
        self.show_metrics = True
        self.scaled_img1 = None # Cached resize
        self.scaled_img2 = None # Cached resize
        self.tk_image = None
        self.slider_pos = 0.5

        self.bind("<Configure>", self.on_resize)
        self.bind("<B1-Motion>", self.on_drag)
        self.bind("<Button-1>", self.on_drag)

    def set_images(self, path1, path2=None, overlay_path1=None, overlay_path2=None, 
                   label1="Right", label2="Left", metrics1="", metrics2="", 
                   show_metrics=True, invert_overlay=True):
        """
        path1: Right Image
        path2: Left Image
        overlay_pathX: Path to error map for blending
        invert_overlay: True for SSIM (brighter=good), False for PSNR/MSE (brighter=bad)
        """
        self.label1 = label1
        self.label2 = label2
        self.metrics1 = metrics1
        self.metrics2 = metrics2
        self.show_metrics = show_metrics
        
        def load_and_blend(img_path, map_path):
            if not img_path or not os.path.exists(img_path):
                return None
            
            img = Image.open(img_path).convert("RGB")
            if not map_path or not os.path.exists(map_path):
                return img
                
            # Perform Blending
            error_map = Image.open(map_path).convert("L")
            if error_map.size != img.size:
                error_map = error_map.resize(img.size, Image.LANCZOS)
                
            import numpy as np
            img_np = np.array(img).astype(np.float32)
            map_np = np.array(error_map).astype(np.float32) / 255.0
            
            # alpha scaling: 0.8 represents greatest error
            if invert_overlay:
                # For SSIM: High value (1.0) = similar, Low value (0.0) = error
                alpha = (1.0 - map_np) * 0.8
            else:
                # For PSNR/MSE: High value (1.0) = error, Low value (0.0) = similar
                alpha = map_np * 0.8
                
            alpha_3d = np.expand_dims(alpha, axis=2)
            
            red = np.zeros_like(img_np)
            red[:, :, 0] = 255.0
            
            blended = img_np * (1.0 - alpha_3d) + red * alpha_3d
            return Image.fromarray(blended.astype(np.uint8))

        try:
            self.image1 = load_and_blend(path1, overlay_path1)
            self.image2 = load_and_blend(path2, overlay_path2)

            self.update_scaled_images()
            self.render()
        except Exception as e:
            print(f"Error loading/blending images: {e}")

    def on_resize(self, event):
        self.update_scaled_images()
        self.render()

    def update_scaled_images(self):
        """Perform the expensive LANCZOS resize only once per window resize or image load."""
        if not self.image1 and not self.image2: return

        # Use the first available image to determine ratio
        base_img = self.image1 if self.image1 else self.image2

        w = self.winfo_width()
        h = self.winfo_height()

        if w < 10 or h < 10: return

        img_w, img_h = base_img.size
        ratio = min(w / img_w, h / img_h)
        self.new_w = int(img_w * ratio)
        self.new_h = int(img_h * ratio)

        if self.image1:
            self.scaled_img1 = self.image1.resize((self.new_w, self.new_h), Image.LANCZOS)
        else:
            self.scaled_img1 = None

        if self.image2:
            self.scaled_img2 = self.image2.resize((self.new_w, self.new_h), Image.LANCZOS)
        else:
            self.scaled_img2 = None

    def on_drag(self, event):
        width = self.winfo_width()
        if width > 0 and self.scaled_img1 and self.scaled_img2: 
            self.slider_pos = max(0, min(1, event.x / width))
            self.render()

    def render(self):
        if not self.scaled_img1 and not self.scaled_img2:
            return

        canvas_width = self.winfo_width()
        canvas_height = self.winfo_height()

        combined = Image.new("RGB", (self.new_w, self.new_h))

        if self.scaled_img1 and self.scaled_img2:
            split_x = int(self.new_w * self.slider_pos)

            left_part = self.scaled_img2.crop((0, 0, split_x, self.new_h))
            right_part = self.scaled_img1.crop((split_x, 0, self.new_w, self.new_h))

            combined.paste(left_part, (0, 0))
            combined.paste(right_part, (split_x, 0))
        elif self.scaled_img1:
            split_x = 0
            combined.paste(self.scaled_img1, (0, 0))
        elif self.scaled_img2:
            split_x = self.new_w
            combined.paste(self.scaled_img2, (0, 0))

        self.tk_image = ImageTk.PhotoImage(combined)
        self.delete("all")

        offset_x = (canvas_width - self.new_w) // 2
        offset_y = (canvas_height - self.new_h) // 2

        self.create_image(offset_x, offset_y, anchor="nw", image=self.tk_image)

        canvas_font = ("sans-serif", 24, "bold")
        metrics_font = ("sans-serif", 14, "bold")

        if self.scaled_img1 and self.scaled_img2:
            line_x = offset_x + split_x
            self.create_line(line_x, offset_y, line_x, offset_y + self.new_h, fill="#00ffff", width=5)

            # Left Label
            self.create_text(offset_x + 20, offset_y + 20, text=self.label2.upper(), fill="#00ffff", anchor="nw", font=canvas_font)
            if self.show_metrics and self.metrics2:
                self.create_text(offset_x + 20, offset_y + 90, text=self.metrics2, fill="#00ffff", anchor="nw", font=metrics_font)

            # Right Label
            self.create_text(offset_x + self.new_w - 20, offset_y + 20, text=self.label1.upper(), fill="#00ffff", anchor="ne", font=canvas_font)
            if self.show_metrics and self.metrics1:
                self.create_text(offset_x + self.new_w - 20, offset_y + 90, text=self.metrics1, fill="#00ffff", anchor="ne", font=metrics_font)

        elif self.scaled_img1:
            self.create_text(offset_x + self.new_w - 20, offset_y + 20, text=f"{self.label1.upper()} (ONLY)", fill="#ffcc00", anchor="ne", font=canvas_font)
            if self.show_metrics and self.metrics1:
                self.create_text(offset_x + self.new_w - 20, offset_y + 90, text=self.metrics1, fill="#ffcc00", anchor="ne", font=metrics_font)
        elif self.scaled_img2:
            self.create_text(offset_x + 20, offset_y + 20, text=f"{self.label2.upper()} (ONLY)", fill="#ffcc00", anchor="nw", font=canvas_font)
            if self.show_metrics and self.metrics2:
                self.create_text(offset_x + 20, offset_y + 90, text=self.metrics2, fill="#ffcc00", anchor="nw", font=metrics_font)
class LICApp:
    def __init__(self, root):
        self.root = root
        self.root.title("LIC Model Visualizer - Desktop")
        self.root.geometry("1600x1000") 
        
        self.zoom_level = 1.0
        self.update_font_scales()
        
        self.apply_styles()
        
        self.registry = self.load_interfaces(os.path.join(ROOT_DIR, "Interfaces", "Testing-Interfaces"))
        self.model_configs = {}
        self.log_queue = queue.Queue()
        self.selected_model_names = []
        self.metrics_data = {} # {model_name: {averages: {}, per_image: []}}
        
        self.setup_ui()
        self.load_metrics() # Added to load existing results on startup
        self.setup_bindings()
        self.poll_log_queue()

    def update_font_scales(self):
        z = self.zoom_level
        self.F_BASE = ("sans-serif", int(14 * z))
        self.F_HEAD = ("sans-serif", int(18 * z), "bold")
        self.F_BTN  = ("sans-serif", int(14 * z), "bold")
        self.F_RUN  = ("sans-serif", int(20 * z), "bold")
        self.F_LOG  = ("monospace", int(12 * z))

    def setup_bindings(self):
        self.root.bind("<Control-plus>", self.zoom_in)
        self.root.bind("<Control-equal>", self.zoom_in)
        self.root.bind("<Control-KP_Add>", self.zoom_in)
        self.root.bind("<Control-minus>", self.zoom_out)
        self.root.bind("<Control-KP_Subtract>", self.zoom_out)
        self.root.bind("<Control-0>", self.zoom_reset)

    def zoom_in(self, event=None):
        self.zoom_level = min(3.0, self.zoom_level + 0.1)
        self.apply_zoom()

    def zoom_out(self, event=None):
        self.zoom_level = max(0.5, self.zoom_level - 0.1)
        self.apply_zoom()

    def zoom_reset(self, event=None):
        self.zoom_level = 1.0
        self.apply_zoom()

    def apply_zoom(self):
        self.update_font_scales()
        self.apply_styles()
        # Trigger UI refresh for some elements if needed, though apply_styles handles most
        self.on_model_select() 

    def apply_styles(self):
        style = ttk.Style()
        if 'clam' in style.theme_names():
            style.theme_use('clam')

        style.configure('.', font=self.F_BASE)
        style.configure('TLabel', font=self.F_BASE)
        style.configure('Header.TLabel', font=self.F_HEAD, foreground="#003366")

        style.configure('TButton', font=self.F_BTN, padding=8)

        style.configure('Run.TButton', font=self.F_RUN, background='#28a745', foreground='white', padding=15)
        style.map('Run.TButton', background=[('active', '#218838')])

        style.configure('TLabelframe.Label', font=self.F_HEAD, foreground="#0055a4")

        style.configure('TCheckbutton', font=self.F_BASE)
        style.configure('TCombobox', font=self.F_BASE)

        style.configure('TEntry', font=self.F_BASE, padding=4, fieldbackground='white')
        style.configure('Treeview', font=self.F_BASE, rowheight=30)
        style.configure('Treeview.Heading', font=self.F_BTN)

    def _on_mousewheel(self, event):
        # Platform-specific mouse wheel handling
        if sys.platform == 'darwin':
            self.config_canvas.yview_scroll(-1 * event.delta, "units")
        else:
            self.config_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def load_interfaces(self, directory):
        registry = {}
        if not os.path.isdir(directory):
            return registry
        for filename in os.listdir(directory):
            if filename.endswith(".py") and not filename.startswith("__"):
                filepath = os.path.join(directory, filename)
                spec = importlib.util.spec_from_file_location(filename[:-3], filepath)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    try:
                        spec.loader.exec_module(module)
                        for name, obj in inspect.getmembers(module, inspect.isclass):
                            if hasattr(obj, 'TASK_NAME') and getattr(obj, 'TASK_NAME') is not None:
                                registry[obj.TASK_NAME] = obj
                    except Exception as e:
                        print(f"Failed to load {filename}: {e}")
        return registry

    def setup_ui(self):
        self.paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.paned.pack(fill=tk.BOTH, expand=True)

        self.sidebar = ttk.Frame(self.paned, width=400, padding=20)
        self.paned.add(self.sidebar, weight=1)

        # 1. Global Toggle at the top
        self.show_advanced_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.sidebar, text="Advanced Mode", variable=self.show_advanced_var, command=self.refresh_sidebar_and_models).pack(anchor="w", pady=(0, 15))

        # 2. Path Settings (GT is always shown)
        self.path_header = ttk.Label(self.sidebar, text="1. Path Settings", style='Header.TLabel')
        self.path_header.pack(anchor="w", pady=(0, 10))

        self.gt_dir_var = tk.StringVar()
        self.gt_label = ttk.Label(self.sidebar, text="Input Images (Ground Truth):", font=self.F_BASE)
        self.gt_label.pack(anchor="w")
        self.gt_frame = ttk.Frame(self.sidebar)
        self.gt_frame.pack(fill=tk.X, pady=(0, 15))
        ttk.Entry(self.gt_frame, textvariable=self.gt_dir_var, font=self.F_BASE).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(self.gt_frame, text="Browse", width=8, command=lambda: self.browse_dir(self.gt_dir_var, check_images=True)).pack(side=tk.LEFT, padx=(5,0))

        # Advanced Global Settings (hidden by default)
        self.adv_global_frame = ttk.Frame(self.sidebar)
        
        self.out_dir_var = tk.StringVar(value=os.path.join(ROOT_DIR, "GUI-Visualizer/outputs"))
        ttk.Label(self.adv_global_frame, text="Results Output:", font=self.F_BASE).pack(anchor="w")
        out_f = ttk.Frame(self.adv_global_frame)
        out_f.pack(fill=tk.X, pady=(0, 15))
        ttk.Entry(out_f, textvariable=self.out_dir_var, font=self.F_BASE).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(out_f, text="Browse", width=8, command=lambda: self.browse_dir(self.out_dir_var)).pack(side=tk.LEFT, padx=(5,0))

        default_base_env = os.path.join(ROOT_DIR, "envs")
        self.base_env_dir_var = tk.StringVar(value=default_base_env if os.path.exists(default_base_env) else "")
        ttk.Label(self.adv_global_frame, text="Environment Folder:", font=self.F_BASE).pack(anchor="w")
        env_f = ttk.Frame(self.adv_global_frame)
        env_f.pack(fill=tk.X, pady=(0, 20))
        ttk.Entry(env_f, textvariable=self.base_env_dir_var, font=self.F_BASE).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(env_f, text="Browse", width=8, command=lambda: self.browse_dir(self.base_env_dir_var)).pack(side=tk.LEFT, padx=(5,0))

        # 3. Model Selection
        ttk.Label(self.sidebar, text="2. Select Models", style='Header.TLabel').pack(anchor="w", pady=(0, 10))

        self.model_listbox = tk.Listbox(
            self.sidebar, 
            selectmode=tk.MULTIPLE, 
            height=6, 
            exportselection=False,
            font=self.F_BASE, 
            bg="#2b2b2b",       
            fg="#ffffff",       
            selectbackground="#007acc", 
            selectforeground="#ffffff",
            highlightthickness=1,
            highlightbackground="#555555"
        )
        for name in sorted(self.registry.keys()):
            self.model_listbox.insert(tk.END, name)
        self.model_listbox.pack(fill=tk.X, pady=(0, 15))
        self.model_listbox.bind("<<ListboxSelect>>", self.on_model_select)

        self.run_btn = ttk.Button(self.sidebar, text="RUN EVALUATION", style='Run.TButton', command=self.run_evaluation)
        self.run_btn.pack(fill=tk.X, pady=15)

        self.progress = ttk.Progressbar(self.sidebar, mode='indeterminate')
        self.progress.pack(fill=tk.X)

        self.main_area = ttk.Notebook(self.paned)
        self.paned.add(self.main_area, weight=4)

        self.config_tab = ttk.Frame(self.main_area, padding=20)
        self.main_area.add(self.config_tab, text="Configuration")

        self.config_canvas = tk.Canvas(self.config_tab, highlightthickness=0)
        self.config_scrollbar = ttk.Scrollbar(self.config_tab, orient="vertical", command=self.config_canvas.yview)
        
        # Static control area for non-dynamic settings
        self.config_static_frame = ttk.Frame(self.config_tab, padding=5)
        self.config_static_frame.pack(fill=tk.X, side=tk.TOP)
        
        self.equalize_var = tk.BooleanVar(value=True)
        self.equalize_check = ttk.Checkbutton(self.config_static_frame, text="Equalize Bitrates", variable=self.equalize_var, command=self.on_equalize_toggle)
        self.equalize_check.pack_forget() # Hidden by default
        
        self.config_scrollable_frame = ttk.Frame(self.config_canvas)
        self.config_scrollable_frame.bind(
            "<Configure>",
            lambda e: self.config_canvas.configure(scrollregion=self.config_canvas.bbox("all"))
        )
        self.config_canvas.create_window((0, 0), window=self.config_scrollable_frame, anchor="nw")
        self.config_canvas.configure(yscrollcommand=self.config_scrollbar.set)

        # Bind mouse wheel for scrolling
        self.config_canvas.bind_all("<MouseWheel>", self._on_mousewheel)

        self.config_canvas.pack(side="left", fill="both", expand=True)
        self.config_scrollbar.pack(side="right", fill="y")

        self.compare_tab = ttk.Frame(self.main_area, padding=15)
        self.main_area.add(self.compare_tab, text="Visual Comparison")

        comp_controls = ttk.Frame(self.compare_tab)
        comp_controls.pack(fill=tk.X, pady=(0, 15))

        ttk.Label(comp_controls, text="Image:", font=self.F_BTN).pack(side=tk.LEFT)
        self.img_selector = ttk.Combobox(comp_controls, state="readonly", width=25, font=self.F_BASE)
        self.img_selector.pack(side=tk.LEFT, padx=(5, 15))
        self.img_selector.bind("<<ComboboxSelected>>", self.update_comparison)

        ttk.Label(comp_controls, text="Left Side:", font=self.F_BTN).pack(side=tk.LEFT)
        self.model_selector_left = ttk.Combobox(comp_controls, state="readonly", width=15, font=self.F_BASE, values=["Ground Truth"])
        self.model_selector_left.set("Ground Truth")
        self.model_selector_left.pack(side=tk.LEFT, padx=5)
        self.model_selector_left.bind("<<ComboboxSelected>>", self.update_comparison)

        ttk.Label(comp_controls, text="Right Side:", font=self.F_BTN).pack(side=tk.LEFT)
        self.model_selector_right = ttk.Combobox(comp_controls, state="readonly", width=15, font=self.F_BASE, values=["Ground Truth"])
        self.model_selector_right.set("Ground Truth")
        self.model_selector_right.pack(side=tk.LEFT, padx=5)
        self.model_selector_right.bind("<<ComboboxSelected>>", self.update_comparison)

        self.equalize_compare_check = ttk.Checkbutton(comp_controls, text="Equalize", variable=self.equalize_var, command=self.on_equalize_toggle)
        # Initially hidden
        
        self.show_metrics_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(comp_controls, text="Show Metrics", variable=self.show_metrics_var, command=self.update_comparison).pack(side=tk.LEFT, padx=15)

        ttk.Label(comp_controls, text="Show Error:", font=self.F_BTN).pack(side=tk.LEFT, padx=(10, 5))
        self.error_type_var = tk.StringVar(value="None")
        self.error_selector = ttk.Combobox(comp_controls, state="readonly", width=8, textvariable=self.error_type_var, values=["None", "PSNR", "SSIM"], font=self.F_BASE)
        self.error_selector.pack(side=tk.LEFT)
        self.error_selector.bind("<<ComboboxSelected>>", self.toggle_error_options)

        self.ssim_overlay_var = tk.BooleanVar(value=True)
        self.ssim_overlay_check = ttk.Checkbutton(comp_controls, text="Overlay", variable=self.ssim_overlay_var, command=self.update_comparison)
        
        self.comp_canvas = ComparisonCanvas(self.compare_tab, bg="#1e1e1e", highlightthickness=0)
        self.comp_canvas.pack(fill=tk.BOTH, expand=True)

        self.metrics_tab = ttk.Frame(self.main_area, padding=20)
        self.main_area.add(self.metrics_tab, text="Metrics Report")

        self.setup_metrics_ui()

        self.log_area = tk.Text(self.sidebar, height=8, font=self.F_LOG, bg="#ffffff", fg="#333333", highlightthickness=1, highlightbackground="#cccccc")
        self.log_area.pack(fill=tk.BOTH, expand=True, pady=(20, 0))
        self.log_area.bind("<Key>", self.block_input)

    def on_equalize_toggle(self):
        self.on_model_select() # Refresh UI to show/hide synchronized QPs

    def sync_qp(self, source_var, *args):
        if not self.equalize_var.get(): return
        try:
            new_val = source_var.get()
        except:
            return
        standard_codecs = ["AVC", "HEVC", "AV1"]
        for mname in self.selected_model_names:
            if mname in standard_codecs:
                qp_var = self.model_configs[mname]["args"].get("qp")
                if qp_var and qp_var.get() != new_val:
                    qp_var.set(new_val)

    def sync_standard_qps(self):
        if not self.equalize_var.get():
            return
        standard_codecs = ["AVC", "HEVC", "AV1"]
        first_qp_var = None
        for mname in self.selected_model_names:
            if mname in standard_codecs:
                qp_var = self.model_configs.get(mname, {}).get("args", {}).get("qp")
                if qp_var:
                    if first_qp_var is None:
                        first_qp_var = qp_var
                    elif qp_var.get() != first_qp_var.get():
                        qp_var.set(first_qp_var.get())

    def toggle_error_options(self, event=None):
        if self.error_type_var.get() != "None":
            self.ssim_overlay_check.pack(side=tk.LEFT, padx=15)
        else:
            self.ssim_overlay_check.pack_forget()
        self.update_comparison()

    def refresh_sidebar_and_models(self):
        """Toggle global path visibility and refresh model config views."""
        if self.show_advanced_var.get():
            # In advanced mode, insert the global adv frame after GT settings
            self.adv_global_frame.pack(after=self.gt_frame, fill=tk.X)
        else:
            self.adv_global_frame.pack_forget()
        
        # Also refresh model configuration tabs
        self.on_model_select()

    def setup_metrics_ui(self):
        self.metrics_notebook = ttk.Notebook(self.metrics_tab)
        self.metrics_notebook.pack(fill=tk.BOTH, expand=True)

        # Tab 1: Single Model Details
        self.model_details_tab = ttk.Frame(self.metrics_notebook, padding=10)
        self.metrics_notebook.add(self.model_details_tab, text="Single Model Details")

        self.metrics_top = ttk.Frame(self.model_details_tab)
        self.metrics_top.pack(fill=tk.X, pady=(0, 15))

        ttk.Label(self.metrics_top, text="Model Performance Summary", style='Header.TLabel').pack(side=tk.LEFT)

        self.metrics_model_sel = ttk.Combobox(self.metrics_top, state="readonly", font=self.F_BASE)
        self.metrics_model_sel.pack(side=tk.RIGHT, padx=5)
        self.metrics_model_sel.bind("<<ComboboxSelected>>", self.refresh_metrics_display)
        ttk.Label(self.metrics_top, text="View Model:", font=self.F_BASE).pack(side=tk.RIGHT)

        self.summary_frame = ttk.LabelFrame(self.model_details_tab, text="Averages", padding=10)
        self.summary_frame.pack(fill=tk.X, pady=(0, 15))
        self.summary_label = ttk.Label(self.summary_frame, text="No evaluation data loaded.", font=self.F_BTN)
        self.summary_label.pack()

        table_frame = ttk.Frame(self.model_details_tab)
        table_frame.pack(fill=tk.BOTH, expand=True)

        columns = ("image", "psnr", "ssim", "lpips", "bpp")
        self.metrics_tree = ttk.Treeview(table_frame, columns=columns, show="headings")

        for col in columns:
            self.metrics_tree.heading(col, text=col.upper())
            self.metrics_tree.column(col, anchor="center", width=120)

        scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=self.metrics_tree.yview)
        self.metrics_tree.configure(yscrollcommand=scrollbar.set)

        self.metrics_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Tab 2: Overall Summary
        self.overall_summary_tab = ttk.Frame(self.metrics_notebook, padding=10)
        self.metrics_notebook.add(self.overall_summary_tab, text="Summary Comparison")

        summary_table_frame = ttk.Frame(self.overall_summary_tab)
        summary_table_frame.pack(fill=tk.BOTH, expand=True)

        sum_cols = ("model", "avg_psnr", "avg_bpp", "best_img", "best_psnr", "worst_img", "worst_psnr", "time")
        self.summary_tree = ttk.Treeview(summary_table_frame, columns=sum_cols, show="headings")
        
        sum_col_widths = {"model": 120, "avg_psnr": 100, "avg_bpp": 80, "best_img": 150, "best_psnr": 100, "worst_img": 150, "worst_psnr": 100, "time": 100}
        for col in sum_cols:
            self.summary_tree.heading(col, text=col.replace("_", " ").upper())
            self.summary_tree.column(col, anchor="center", width=sum_col_widths.get(col, 100))

        sum_scroll = ttk.Scrollbar(summary_table_frame, orient="vertical", command=self.summary_tree.yview)
        self.summary_tree.configure(yscrollcommand=sum_scroll.set)
        self.summary_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sum_scroll.pack(side=tk.RIGHT, fill=tk.Y)

    def block_input(self, event):
        is_modifier = event.state & (0x4 | 0x8 | 0x10 | 0x40)
        if is_modifier and event.keysym.lower() in ('c', 'a'):
            return None
        if event.keysym in ('Left', 'Right', 'Up', 'Down', 'Prior', 'Next', 'Home', 'End'):
            return None
        return "break"

    def poll_log_queue(self):
        try:
            while True:
                msg = self.log_queue.get_nowait()
                self.log_area.insert(tk.END, msg)
                self.log_area.see(tk.END)
        except queue.Empty:
            pass
        self.root.after(100, self.poll_log_queue)

    def browse_dir(self, var, check_images=False):
        current = var.get()
        initial = current if current and os.path.exists(current) else ROOT_DIR
        path = filedialog.askdirectory(initialdir=initial)
        if path:
            path = os.path.abspath(os.path.expanduser(path))
            if check_images:
                try:
                    has_images = any(f.lower().endswith(('.png', '.jpg', '.jpeg')) for f in os.listdir(path))
                    if not has_images:
                        messagebox.showwarning("Warning", "The selected directory appears to have no images.")
                except Exception as e:
                    self.log(f"[ERROR] Could not read directory {path}: {e}\n")

            var.set(path)
            if check_images:
                self.refresh_image_list()
            
            # Reload metrics if the output directory changed
            if var == self.out_dir_var:
                self.load_metrics()

    def browse_file(self, var):
        current = var.get()
        initial = os.path.dirname(current) if current and os.path.exists(os.path.dirname(current)) else ROOT_DIR
        path = filedialog.askopenfilename(initialdir=initial)
        if path:
            var.set(os.path.abspath(os.path.expanduser(path)))
    def log(self, msg):
        self.log_queue.put(msg)

    def on_model_select(self, event=None):
        for widget in self.config_scrollable_frame.winfo_children():
            widget.destroy()

        selected_indices = self.model_listbox.curselection()
        self.selected_model_names = [self.model_listbox.get(i) for i in selected_indices]

        # Populate both comparison selectors with GT + selected models
        comp_values = ["Ground Truth"] + self.selected_model_names
        self.model_selector_left['values'] = comp_values
        self.model_selector_right['values'] = comp_values
        
        # Set defaults: Left=GT, Right=First Model (if available)
        self.model_selector_left.set("Ground Truth")
        if self.selected_model_names:
            self.model_selector_right.set(self.selected_model_names[0])
        else:
            self.model_selector_right.set("Ground Truth")

        self.metrics_model_sel['values'] = self.selected_model_names

        # Toggle equalize visibility
        standard_codecs = ["AVC", "HEVC", "AV1"]
        selected_standards = [m for m in self.selected_model_names if m in standard_codecs]
        if len(selected_standards) >= 2:
            self.equalize_check.pack(side=tk.TOP, pady=5)
        else:
            self.equalize_check.pack_forget()

        for name in self.selected_model_names:
            self.build_model_config_ui(name)

        self.sync_standard_qps()
        
        self.update_comparison()

    def build_model_config_ui(self, model_name):
        frame = ttk.LabelFrame(self.config_scrollable_frame, text=f"Config: {model_name}", padding=15)
        frame.pack(fill=tk.X, pady=10, padx=5)

        interface_cls = self.registry[model_name]
        required_args = getattr(interface_cls, 'REQUIRED_ARGS', [])
        show_advanced = self.show_advanced_var.get()
        standard_codecs = ["AVC", "HEVC", "AV1"]
        
        if model_name not in self.model_configs:
            # Set workdir to LIC-Models for standard codecs, otherwise LIC-Models/Name
            workdir_val = "LIC-Models" if model_name in standard_codecs else f"LIC-Models/{model_name}"

            self.model_configs[model_name] = {
                "args": {},
                "workdir": tk.StringVar(value=workdir_val),
                "env": tk.StringVar()
            }

            defaults = getattr(interface_cls, 'DEFAULT_VARS', {})
            for k, v in defaults.items():
                if k in ['data', 'dataset', 'input', 'input_dir', 'save_dir', 'output']: continue
                var = tk.StringVar(value="" if v is None else str(v))
                if model_name in standard_codecs and k == "qp":
                    var.trace_add("write", lambda *args, v=var: self.sync_qp(v, *args))
                self.model_configs[model_name]["args"][k] = var
            for req in required_args:
                if req in ['data', 'dataset', 'input', 'input_dir', 'save_dir', 'output']: continue
                if req not in self.model_configs[model_name]["args"]:
                    self.model_configs[model_name]["args"][req] = tk.StringVar()

        config = self.model_configs[model_name]

        def create_row(label_text, var, is_dir=False, is_file=False, required=False):
            row = ttk.Frame(frame)
            row.pack(fill=tk.X, pady=5)

            display_text = label_text
            if required:
                display_text = "* " + display_text

            lbl = ttk.Label(row, text=display_text, width=20, font=self.F_BTN)
            lbl.pack(side=tk.LEFT)
            if required:
                lbl.configure(foreground="#cc0000")

            entry = ttk.Entry(row, textvariable=var, font=self.F_BASE)
            entry.pack(side=tk.LEFT, fill=tk.X, expand=True)

            if is_dir or is_file:
                ttk.Button(row, text="Browse", width=8, command=lambda: self.browse_dir(var) if is_dir else self.browse_file(var)).pack(side=tk.LEFT, padx=(5,0))

        if show_advanced:
            create_row("Working Dir:", config["workdir"], is_dir=True, required=True)
            create_row("Env Path:", config["env"], is_dir=True)
            ttk.Separator(frame, orient='horizontal').pack(fill=tk.X, pady=10)

        for arg_name, var in config["args"].items():
            # Filter what to show in non-advanced mode
            if not show_advanced:
                # For standard codecs, only show QP and Use GPU
                if model_name in ["AVC", "HEVC", "AV1"]:
                    if arg_name.lower() not in ["qp", "use_gpu"]:
                        continue
                else:
                    # Always hide things that are standard defaults or handled elsewhere
                    if arg_name.lower() in ['cuda', 'experiment', 'test_batch_size', 'num_workers', 'real', 'clip_max_norm', 'save_dir', 'output', 'train_dataset', 'test_dataset']:
                        continue

                    # Only show if required, or if it doesn't have a default value in DEFAULT_VARS (meaning it's essential to provide)
                    is_required = arg_name in required_args
                    has_default = arg_name in getattr(interface_cls, 'DEFAULT_VARS', {})

                    # If it's not required and has a default, skip it in simple mode
                    if not is_required and has_default:
                        continue
            arg_lower = arg_name.lower()
            is_file = any(x in arg_lower for x in ['checkpoint', 'model', 'file', 'elic', 'codec', 'pth', 'pkl', 'weights'])
            is_dir = any(x in arg_lower for x in ['dir', 'dataset', 'folder', 'save'])
            if 'path' in arg_lower and not is_file and not is_dir:
                is_dir = True
            
            is_required = arg_name in required_args
            
            # Autofill for checkpoints/models if empty
            if is_file and (not var.get() or var.get() == "None"):
                workdir = os.path.join(ROOT_DIR, config["workdir"].get())
                # Try common locations
                search_dirs = [
                    workdir,
                    os.path.join(workdir, "checkpoints"),
                    os.path.join(ROOT_DIR, "checkpoints")
                ]
                found_files = []
                for sd in search_dirs:
                    if os.path.exists(sd):
                        for ext in ["*.pth", "*.pth.tar", "*.pkl", "*.pt"]:
                            for depth in range(4):
                                wildcards = ["*"] * depth
                                pattern = os.path.join(sd, *wildcards, ext)
                                found_files.extend(glob.glob(pattern))
                
                if found_files:
                    # Priority: 1. Contains 'best', 2. Newest modified
                    best_files = [f for f in found_files if 'best' in os.path.basename(f).lower()]
                    if best_files:
                        best_files.sort(key=os.path.getmtime, reverse=True)
                        var.set(best_files[0])
                    else:
                        found_files.sort(key=os.path.getmtime, reverse=True)
                        var.set(found_files[0])

            create_row(f"{arg_name}:", var, is_dir=is_dir, is_file=is_file, required=is_required)

    def run_evaluation(self):
        if not self.selected_model_names:
            messagebox.showerror("Error", "Please select at least one model from the Models list.")
            return
        gt_dir = os.path.expanduser(self.gt_dir_var.get().strip())
        if not gt_dir or not os.path.isdir(gt_dir):
            messagebox.showerror("Error", f"Valid GT directory required.\nCurrent path: {gt_dir}")
            return
        self.run_btn.config(state=tk.DISABLED)
        self.progress.start()
        threading.Thread(target=self.execution_thread, args=(gt_dir,), daemon=True).start()

    def execution_thread(self, gt_dir):
        base_env_dir = self.base_env_dir_var.get().strip()
        
        def find_env(name):
            # Try user specified base
            if base_env_dir and os.path.exists(os.path.join(base_env_dir, name)):
                return os.path.join(base_env_dir, name)
            # Try project-relative LIC-Models
            if os.path.exists(os.path.join(ROOT_DIR, "LIC-Models", name)):
                return os.path.join(ROOT_DIR, "LIC-Models", name)
            # Try project-relative root
            if os.path.exists(os.path.join(ROOT_DIR, name)):
                return os.path.join(ROOT_DIR, name)
            # Try envs/ directory
            if os.path.exists(os.path.join(ROOT_DIR, "envs", name)):
                return os.path.join(ROOT_DIR, "envs", name)
            return None

        output_base = os.path.expanduser(self.out_dir_var.get().strip())
        standard_codecs = ["AVC", "HEVC", "AV1"]
        selected_standards = [m for m in self.selected_model_names if m in standard_codecs]
        learning_models = [m for m in self.selected_model_names if m not in standard_codecs]
        equalize_enabled = self.equalize_var.get() and len(selected_standards) >= 2
        tasks = {}
        eval_tasks = []
        standard_equalize_config = None
        standard_use_gpu = None
        for model_name in self.selected_model_names:
            config = self.model_configs[model_name]
            final_args = {}
            for k, v in config["args"].items():
                val = v.get()
                if isinstance(val, str) and (os.path.sep in val or val.endswith(('.pth', '.pt', '.pkl', 'sd-turbo', 'elic.pth'))):
                     val = os.path.abspath(os.path.join(ROOT_DIR, val))
                final_args[k] = val
            interface_cls = self.registry[model_name]
            data_keys = ['input', 'input_dir', 'data', 'dataset', 'test_dataset', 'train_dataset']
            target_key = next((k for k in data_keys if k in interface_cls.REQUIRED_ARGS or k in getattr(interface_cls, 'DEFAULT_VARS', {})), "dataset")
            final_args[target_key] = gt_dir
            model_out = os.path.abspath(os.path.join(output_base, model_name))
            os.makedirs(model_out, exist_ok=True)

#             if model_name == "LIC-HPCM": final_args["save_dir"] = model_out
#             elif model_name == "StableCodec":
#                 final_args["rec_path"] = model_out
#                 final_args["bin_path"] = os.path.join(model_out, "bins")
#             elif model_name == "DCVC-RT": final_args["save_dir"] = model_out
#             elif model_name == "ELIC": final_args["experiment"] = f"gui_eval_{model_name}"
            if model_name == "RwkvCompress": 
                final_args["save_dir"] = model_out
                # Ensure only ONE quality is used in GUI mode for compatibility
                if "qualities" in final_args:
                    qs = final_args["qualities"]
                    if isinstance(qs, str):
                        qs = qs.split()
                    if isinstance(qs, list) and len(qs) > 1:
                        self.log(f"[GUI] Warning: RwkvCompress supports multiple qualities, but GUI only visualizes one. Using first quality: {qs[0]}\n")
                        final_args["qualities"] = [qs[0]]
                    if "checkpoints" in final_args:
                        ckpts = final_args["checkpoints"]
                        if isinstance(ckpts, str):
                            ckpts = ckpts.split()
                        if isinstance(ckpts, list) and len(ckpts) > 1:
                            final_args["checkpoints"] = [ckpts[0]]
            else:
              final_args["save_dir"] = model_out

            # Derive or use custom model environment path
            user_model_env = config["env"].get().strip()
            if user_model_env:
                model_env = os.path.abspath(os.path.expanduser(user_model_env))
            else:
                model_env = find_env(f"{model_name}-env")

            if not model_env or not os.path.exists(model_env):
                self.log(f"[WARNING] Environment for {model_name} not found. Looked in common project locations.\n")
                if not self.show_advanced_var.get():
                     self.log("[HINT] Switch to 'Advanced Mode' to manually set the environment path if it is in a non-standard location.\n")

            tasks[model_name] = {
                "task_name": model_name,
                "directory": os.path.join(ROOT_DIR, config["workdir"].get()),
                "env_path": model_env if model_env and os.path.exists(model_env) else None,
                "arguments": final_args
            }
            eval_tasks.append({
                "task_name": model_name,
                "save_dir": model_out,
                "input_dir": gt_dir
            })

            if model_name in standard_codecs and equalize_enabled:
                if standard_equalize_config is None:
                    standard_equalize_config = {
                        "codec_list": ",".join(selected_standards),
                        "qp": final_args.get("qp"),
                        "use_gpu": final_args.get("use_gpu"),
                        "input_dir": gt_dir,
                        "save_dir": model_out,
                        "workdir": os.path.join(ROOT_DIR, config["workdir"].get()),
                        "env_path": model_env if model_env and os.path.exists(model_env) else None
                    }
                    standard_use_gpu = final_args.get("use_gpu")
                else:
                    if standard_use_gpu != final_args.get("use_gpu"):
                        self.log("[GUI] Warning: Equalize enabled, but standard codecs have different 'use_gpu' values. Using the first codec's setting.\n")

        # Derive eval environment path
        eval_env_path = find_env("eval-env")
        if not eval_env_path or not os.path.exists(eval_env_path):
             self.log(f"[WARNING] Evaluation environment 'eval-env' not found.\n")
             if not self.show_advanced_var.get():
                self.log("[HINT] Evaluation requires 'evaluation.py' to run in a specific environment. Ensure it was created via quick-start-env.py or set it in Advanced Mode.\n")

        # Track total run time per model
        model_timings = {}

        original_print = builtins.print
        def custom_print(*args, **kwargs):
            msg = " ".join(map(str, args))
            end = kwargs.get('end', '\n')
            self.log(msg + end)
        builtins.print = custom_print

        original_input = builtins.input
        def custom_input(prompt=""):
            self.log(f"{prompt} [Auto-skipping invalid path]\n")
            return "skip"
        builtins.input = custom_input

        orig_run = subprocess.run
        orig_check_call = subprocess.check_call

        def streaming_executor(cmd, **kwargs):
            kwargs.pop('check', None)
            kwargs.pop('capture_output', None)
            kwargs['stdout'] = subprocess.PIPE
            kwargs['stderr'] = subprocess.STDOUT
            kwargs['text'] = True
            kwargs['bufsize'] = 1
            kwargs['universal_newlines'] = True
            process = subprocess.Popen(cmd, **kwargs)
            if process.stdout:
                for line in process.stdout:
                    self.log(line)
            return process.wait()

        def custom_check_call(cmd, *args, **kwargs):
            ret = streaming_executor(cmd, **kwargs)
            if ret != 0: raise subprocess.CalledProcessError(ret, cmd)
            return 0

        def custom_run(cmd, *args, **kwargs):
            ret = streaming_executor(cmd, **kwargs)
            return subprocess.CompletedProcess(cmd, ret)

        subprocess.run = custom_run
        subprocess.check_call = custom_check_call


        def build_learning_target_map():
            target_map = {}
            for model_name in learning_models:
                metrics_path = os.path.join(output_base, model_name, f"{model_name}_metrics.json")
                if not os.path.exists(metrics_path):
                    self.log(f"[GUI] Warning: Metrics not found for {model_name}; skipping as target source.\n")
                    continue
                try:
                    with open(metrics_path, "r") as f:
                        data = json.load(f)
                except Exception as e:
                    self.log(f"[GUI] Warning: Failed to read metrics for {model_name}: {e}\n")
                    continue
                for item in data.get("per_image_metrics", []):
                    img_name = item.get("image_name")
                    bpp_val = item.get("bpp")
                    if not img_name or not isinstance(bpp_val, (int, float)):
                        continue
                    base = os.path.splitext(img_name)[0]
                    if base not in target_map or bpp_val < target_map[base]:
                        target_map[base] = bpp_val
            return target_map

        try:
            standard_equalized = False
            standard_elapsed = 0.0
            target_bpp_json = None
            ordered_models = list(self.selected_model_names)
            if equalize_enabled and learning_models:
                ordered_models = learning_models + [m for m in self.selected_model_names if m in standard_codecs]

            for model_name in ordered_models:
                task_info = tasks[model_name]
                start_t = time.time()
                self.log(f"\n[GUI] Starting Task: {model_name}...\n")

                if equalize_enabled and model_name in standard_codecs:
                    if not standard_equalized:
                        if not standard_equalize_config:
                            self.log("[GUI ERROR] Equalize enabled, but no standard codec config was found.\n")
                            raise RuntimeError("Missing standard codec config for equalization")

                        if learning_models and not target_bpp_json:
                            target_map = build_learning_target_map()
                            if target_map:
                                target_bpp_json = os.path.join(output_base, "standard_equalize_targets.json")
                                with open(target_bpp_json, "w") as f:
                                    json.dump(target_map, f, indent=4)
                                self.log("[GUI] Using learning-based bpp targets for standard codec equalization.\n")
                            else:
                                self.log("[GUI] Warning: No learning-based bpp targets found; using standard anchor only.\n")
                        
                        python_exec = sys.executable
                        env_path = standard_equalize_config.get("env_path")
                        if env_path:
                            if os.path.isdir(os.path.join(env_path, "Scripts")):
                                python_exec = os.path.join(env_path, "Scripts", "python.exe")
                            else:
                                python_exec = os.path.join(env_path, "bin", "python3")

                        cmd = [
                            python_exec,
                            os.path.join(ROOT_DIR, "LIC-Models", "Standard-Codecs", "eval_standard.py"),
                            "--codec", standard_equalize_config["codec_list"],
                            "--qp", str(standard_equalize_config["qp"]),
                            "--input_dir", standard_equalize_config["input_dir"],
                            "--save_dir", standard_equalize_config["save_dir"]
                        ]
                        if standard_equalize_config.get("use_gpu"):
                            cmd.append("--use_gpu")
                        if target_bpp_json:
                            cmd.extend(["--target_bpp_json", target_bpp_json])
                        cmd.append("--equalize")

                        self.log(f"[GUI] Equalizing standard codecs: {standard_equalize_config['codec_list']}\n")
                        subprocess.run(cmd, cwd=standard_equalize_config.get("workdir"))
                        standard_elapsed = time.time() - start_t
                        standard_equalized = True

                    # Run evaluation for this specific model immediately to get metrics
                    this_eval_env = eval_env_path if os.path.exists(eval_env_path) else "n/a"
                    python_exec = sys.executable
                    if this_eval_env != "n/a":
                        if os.path.isdir(os.path.join(this_eval_env, "Scripts")):
                            python_exec = os.path.join(this_eval_env, "Scripts", "python.exe")
                        else:
                            python_exec = os.path.join(this_eval_env, "bin", "python3")

                    eval_job = next(j for j in eval_tasks if j["task_name"] == model_name)
                    cmd = [
                        python_exec, os.path.join(ROOT_DIR, "evaluation.py"),
                        "--task_name", model_name,
                        "--save_dir", eval_job["save_dir"],
                        "--input_dir", eval_job["input_dir"]
                    ]
                    subprocess.run(cmd)

                    model_timings[model_name] = standard_elapsed

                    # Update the metrics file with timing info
                    metrics_file = os.path.join(eval_job["save_dir"], f"{model_name}_metrics.json")
                    if os.path.exists(metrics_file):
                        with open(metrics_file, 'r') as f:
                            m_data = json.load(f)
                        m_data["runtime_seconds"] = standard_elapsed
                        with open(metrics_file, 'w') as f:
                            json.dump(m_data, f, indent=4)

                    continue

                # Create a temporary single-task config for the dispatcher
                temp_config = {
                    "testing": {
                        "tasks": {model_name: task_info}
                    }
                }
                temp_path = os.path.join(ROOT_DIR, f"gui_temp_{model_name}.json")
                with open(temp_path, 'w') as f:
                    json.dump(temp_config, f, indent=4)
                
                dispatcher = Dispatcher(
                    arg_json_path=temp_path,
                    run_test=True,
                    test_interfaces_path=os.path.join(ROOT_DIR, "Interfaces", "Testing-Interfaces")
                )
                dispatcher.run()
                
                # Run evaluation for this specific model immediately to get metrics
                this_eval_env = eval_env_path if os.path.exists(eval_env_path) else "n/a"
                python_exec = sys.executable
                if this_eval_env != "n/a":
                    if os.path.isdir(os.path.join(this_eval_env, "Scripts")):
                        python_exec = os.path.join(this_eval_env, "Scripts", "python.exe")
                    else:
                        python_exec = os.path.join(this_eval_env, "bin", "python3")
                
                eval_job = next(j for j in eval_tasks if j["task_name"] == model_name)
                cmd = [
                    python_exec, os.path.join(ROOT_DIR, "evaluation.py"),
                    "--task_name", model_name,
                    "--save_dir", eval_job["save_dir"],
                    "--input_dir", eval_job["input_dir"]
                ]
                subprocess.run(cmd)
                
                elapsed = time.time() - start_t
                model_timings[model_name] = elapsed
                
                # Update the metrics file with timing info
                metrics_file = os.path.join(eval_job["save_dir"], f"{model_name}_metrics.json")
                if os.path.exists(metrics_file):
                    with open(metrics_file, 'r') as f:
                        m_data = json.load(f)
                    m_data["runtime_seconds"] = elapsed
                    with open(metrics_file, 'w') as f:
                        json.dump(m_data, f, indent=4)
                
                if os.path.exists(temp_path):
                    os.remove(temp_path)

            self.log("\n[GUI] All tasks completed successfully.\n")

        except Exception as e:
            self.log(f"\n[GUI ERROR] Dispatcher failed: {e}\n")
        finally:
            builtins.print = original_print
            builtins.input = original_input
            subprocess.run = orig_run
            subprocess.check_call = orig_check_call

        self.root.after(0, self.finish_execution)

    def finish_execution(self):
        self.progress.stop()
        self.run_btn.config(state=tk.NORMAL)
        self.load_metrics()
        self.refresh_image_list()
        messagebox.showinfo("Done", "Evaluation complete.")

    def load_metrics(self):
        output_base = os.path.expanduser(self.out_dir_var.get().strip())
        self.metrics_data = {}
        
        if not os.path.exists(output_base):
            return

        # Scan all subdirectories in the output base for metrics files
        for model_folder in os.listdir(output_base):
            folder_path = os.path.join(output_base, model_folder)
            if os.path.isdir(folder_path):
                metrics_file = os.path.join(folder_path, f"{model_folder}_metrics.json")
                if os.path.exists(metrics_file):
                    try:
                        with open(metrics_file, 'r') as f:
                            self.metrics_data[model_folder] = json.load(f)
                    except Exception as e:
                        self.log(f"[ERROR] Failed to load metrics for {model_folder}: {e}\n")
        
        if self.metrics_data:
            available_models = sorted(list(self.metrics_data.keys()))
            self.metrics_model_sel['values'] = available_models
            if not self.metrics_model_sel.get() or self.metrics_model_sel.get() not in self.metrics_data:
                self.metrics_model_sel.set(available_models[0])
            self.refresh_metrics_display()

    def refresh_summary_report(self):
        for item in self.summary_tree.get_children():
            self.summary_tree.delete(item)
            
        for model_name, data in self.metrics_data.items():
            avg = data.get("averages", {})
            runtime = data.get("runtime_seconds", 0)
            per_img = data.get("per_image_metrics", [])
            
            if not per_img: continue
            
            # Find best/worst based on PSNR
            best = max(per_img, key=lambda x: x.get("psnr", 0))
            worst = min(per_img, key=lambda x: x.get("psnr", 0))
            
            time_str = f"{runtime:.1f}s" if runtime < 60 else f"{runtime/60:.1f}m"
            
            self.summary_tree.insert("", tk.END, values=(
                model_name,
                avg.get("psnr"),
                avg.get("bpp"),
                best.get("image_name"),
                best.get("psnr"),
                worst.get("image_name"),
                worst.get("psnr"),
                time_str
            ))

    def refresh_metrics_display(self, event=None):
        model_name = self.metrics_model_sel.get()
        if not model_name or model_name not in self.metrics_data:
            return
        data = self.metrics_data[model_name]
        avg = data.get("averages", {})
        runtime = data.get("runtime_seconds", "N/A")
        time_str = f"{runtime:.2f}s" if isinstance(runtime, (int, float)) else "N/A"
        summary_text = f"PSNR: {avg.get('psnr', 'N/A')} dB | SSIM: {avg.get('ssim', 'N/A')} | LPIPS: {avg.get('lpips', 'N/A')} | bpp: {avg.get('bpp', 'N/A')} | Time: {time_str}"
        self.summary_label.config(text=summary_text)
        
        for item in self.metrics_tree.get_children():
            self.metrics_tree.delete(item)
        for item in data.get("per_image_metrics", []):
            self.metrics_tree.insert("", tk.END, values=(
                item.get("image_name"),
                item.get("psnr"),
                item.get("ssim"),
                item.get("lpips"),
                item.get("bpp")
            ))
        
        self.refresh_summary_report()

    def refresh_image_list(self):
        gt_dir = os.path.expanduser(self.gt_dir_var.get().strip())
        if os.path.isdir(gt_dir):
            imgs = sorted([os.path.basename(f) for f in glob.glob(os.path.join(gt_dir, "*")) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            self.img_selector['values'] = imgs
            if imgs: self.img_selector.current(0)

    def get_model_image_path(self, model_name, selected_img, subfolder="reconstruction"):
        if model_name == "Ground Truth":
            gt_dir = os.path.expanduser(self.gt_dir_var.get().strip())
            return os.path.join(gt_dir, selected_img)
            
        out_base = os.path.expanduser(self.out_dir_var.get().strip())
        model_out = os.path.join(out_base, model_name)
        if not os.path.exists(model_out):
            return None
            
        base_name = os.path.splitext(selected_img)[0]
        
        # 1. Check for standard subfolder (reconstruction, ssim_map)
        target = os.path.join(model_out, subfolder)
        if os.path.exists(target):
            for f in os.listdir(target):
                if (f == selected_img or f.startswith(base_name)) and f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    return os.path.join(target, f)
        
        # 2. Check for nested quality subfolders (RwkvCompress)
        for q_dir in os.listdir(model_out):
            if q_dir.startswith("quality_"):
                target = os.path.join(model_out, q_dir, subfolder)
                if os.path.exists(target):
                    for f in os.listdir(target):
                        if (f == selected_img or f.startswith(base_name)) and f.lower().endswith(('.png', '.jpg', '.jpeg')):
                            return os.path.join(target, f)
                            
        # 3. Fallback to model root for reconstruction
        if subfolder == "reconstruction":
            for f in os.listdir(model_out):
                if (f == selected_img or f.startswith(base_name)) and f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    return os.path.join(model_out, f)
        
        return None

    def get_metrics_string(self, model_name, selected_img, image_path):
        if model_name == "Ground Truth":
            if image_path and os.path.exists(image_path):
                try:
                    with Image.open(image_path) as img:
                        w, h = img.size
                    size_bits = os.path.getsize(image_path) * 8
                    bpp = size_bits / (w * h)
                    return f"bpp: {bpp:.4f}"
                except:
                    return ""
            return ""
            
        if model_name in self.metrics_data:
            data = self.metrics_data[model_name]
            # Match by full name or basename
            base = os.path.splitext(selected_img)[0]
            for item in data.get("per_image_metrics", []):
                iname = item.get("image_name", "")
                if iname == selected_img or os.path.splitext(iname)[0] == base:
                    psnr = item.get("psnr", "N/A")
                    py = item.get("psnr_y", "N/A")
                    pu = item.get("psnr_u", "N/A")
                    pv = item.get("psnr_v", "N/A")
                    ssim = item.get("ssim", "N/A")
                    bpp = item.get("bpp", "N/A")
                    qp = item.get("qp", None)
                    
                    line1 = f"PSNR: {psnr} | Y: {py} | U: {pu} | V: {pv}"
                    line2 = f"SSIM: {ssim} | bpp: {bpp}"
                    if qp not in (None, "", "N/A"):
                        line2 += f" | qp: {qp}"
                    return f"{line1}\n{line2}"
        return ""

    def update_comparison(self, event=None):
        selected_img = self.img_selector.get()
        model_left = self.model_selector_left.get()
        model_right = self.model_selector_right.get()

        if not selected_img or not model_left or not model_right: 
            return

        error_type = self.error_type_var.get()
        use_overlay = self.ssim_overlay_var.get()
        
        show_error = (error_type != "None")
        invert_overlay = (error_type == "SSIM")
        
        # Determine base paths and overlay paths
        subfolder = "reconstruction"
        if show_error:
            if error_type == "PSNR": subfolder = "psnr_map"
            elif error_type == "SSIM": subfolder = "ssim_map"

        if show_error and use_overlay:
            # Base is reconstruction, overlay is the error map
            path_left = self.get_model_image_path(model_left, selected_img, "reconstruction")
            path_right = self.get_model_image_path(model_right, selected_img, "reconstruction")
            overlay_left = self.get_model_image_path(model_left, selected_img, subfolder) if model_left != "Ground Truth" else None
            overlay_right = self.get_model_image_path(model_right, selected_img, subfolder) if model_right != "Ground Truth" else None
        elif show_error:
            # Base is the error map directly
            path_left = self.get_model_image_path(model_left, selected_img, subfolder)
            path_right = self.get_model_image_path(model_right, selected_img, subfolder)
            overlay_left = overlay_right = None
        else:
            # Standard reconstruction view
            path_left = self.get_model_image_path(model_left, selected_img, "reconstruction")
            path_right = self.get_model_image_path(model_right, selected_img, "reconstruction")
            overlay_left = overlay_right = None

        show_m = self.show_metrics_var.get()
        metrics_left = self.get_metrics_string(model_left, selected_img, path_left)
        metrics_right = self.get_metrics_string(model_right, selected_img, path_right)

        self.comp_canvas.set_images(
            path_right, path_left, 
            overlay_path1=overlay_right, overlay_path2=overlay_left,
            label1=model_right, label2=model_left,
            metrics1=metrics_right, metrics2=metrics_left,
            show_metrics=show_m,
            invert_overlay=invert_overlay
        )

        log_msg = f"[Viewer] Comparing: {model_left} vs {model_right} ({selected_img})"
        if show_error: log_msg += f" [{error_type} Map]"
        if show_error and use_overlay: log_msg += " [Overlay Mode]"
        self.log(log_msg + "\n")

if __name__ == "__main__":
    root = tk.Tk()
    
    # Environment-based scaling
    scaling_factor = 2.0 # Default for Mac/Retina
    
    try:
        # Check if running in WSL
        is_wsl = False
        if sys.platform == "linux":
            try:
                with open("/proc/version", "r") as f:
                    if "microsoft" in f.read().lower():
                        is_wsl = True
            except:
                pass
        
        if is_wsl:
            # WSL 4K usually needs much higher scaling (3.0 - 4.0)
            scaling_factor = 3.5
            print(f"[INFO] WSL detected, applying scaling factor: {scaling_factor}")
        elif sys.platform == "darwin":
            # Mac scaling is already good at 2.0
            scaling_factor = 2.0
            
        root.tk.call('tk', 'scaling', scaling_factor)
    except Exception as e:
        print(f"[WARNING] Could not set scaling: {e}")
        
    app = LICApp(root)
    
    # Maximize window on startup
    if sys.platform == "win32":
        root.state('zoomed')
    else:
        root.attributes('-zoomed', True)
        
    root.mainloop()
