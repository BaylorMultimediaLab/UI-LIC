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
        self.image1 = None # GT
        self.image2 = None # Compressed
        self.scaled_img1 = None # Cached resize
        self.scaled_img2 = None # Cached resize
        self.tk_image = None
        self.slider_pos = 0.5
        
        self.bind("<Configure>", self.on_resize)
        self.bind("<B1-Motion>", self.on_drag)
        self.bind("<Button-1>", self.on_drag)

    def set_images(self, path1, path2=None):
        try:
            self.image1 = Image.open(path1).convert("RGB")
            if path2 and os.path.exists(path2):
                self.image2 = Image.open(path2).convert("RGB")
            else:
                self.image2 = None
                
            self.update_scaled_images()
            self.render()
        except Exception as e:
            print(f"Error loading images: {e}")

    def on_resize(self, event):
        self.update_scaled_images()
        self.render()

    def update_scaled_images(self):
        """Perform the expensive LANCZOS resize only once per window resize or image load."""
        if not self.image1: return
        
        w = self.winfo_width()
        h = self.winfo_height()
        
        if w < 10 or h < 10: return
        
        img_w, img_h = self.image1.size
        ratio = min(w / img_w, h / img_h)
        self.new_w = int(img_w * ratio)
        self.new_h = int(img_h * ratio)
        
        self.scaled_img1 = self.image1.resize((self.new_w, self.new_h), Image.LANCZOS)
        
        if self.image2:
            self.scaled_img2 = self.image2.resize((self.new_w, self.new_h), Image.LANCZOS)
        else:
            self.scaled_img2 = None

    def on_drag(self, event):
        width = self.winfo_width()
        if width > 0 and self.scaled_img2: 
            self.slider_pos = max(0, min(1, event.x / width))
            self.render()

    def render(self):
        if not getattr(self, 'scaled_img1', None):
            return

        canvas_width = self.winfo_width()
        canvas_height = self.winfo_height()
        
        combined = Image.new("RGB", (self.new_w, self.new_h))
        
        if self.scaled_img2:
            split_x = int(self.new_w * self.slider_pos)
            
            left_part = self.scaled_img2.crop((0, 0, split_x, self.new_h))
            right_part = self.scaled_img1.crop((split_x, 0, self.new_w, self.new_h))
            
            combined.paste(left_part, (0, 0))
            combined.paste(right_part, (split_x, 0))
        else:
            split_x = 0
            combined.paste(self.scaled_img1, (0, 0))

        self.tk_image = ImageTk.PhotoImage(combined)
        self.delete("all")
        
        offset_x = (canvas_width - self.new_w) // 2
        offset_y = (canvas_height - self.new_h) // 2
        
        self.create_image(offset_x, offset_y, anchor="nw", image=self.tk_image)
        
        # Scale canvas font
        # Trying to find the app instance to get scale_factor, or just use a relative size
        # Better: we'll pass scale_factor to ComparisonCanvas if needed, 
        # but for now we'll use a heuristic.
        scale = getattr(self.master, 'scale_factor', 1.0)
        canvas_font = ("sans-serif", int(24 * scale), "bold")
        
        if self.scaled_img2:
            line_x = offset_x + split_x
            self.create_line(line_x, offset_y, line_x, offset_y + self.new_h, fill="#00ffff", width=int(5 * scale))
            self.create_text(offset_x + 20*scale, offset_y + 20*scale, text="COMPRESSED", fill="#00ffff", anchor="nw", font=canvas_font)
            self.create_text(offset_x + self.new_w - 20*scale, offset_y + 20*scale, text="GROUND TRUTH", fill="#00ffff", anchor="ne", font=canvas_font)
        else:
            self.create_text(offset_x + self.new_w - 20*scale, offset_y + 20*scale, text="GROUND TRUTH (Awaiting Output...)", fill="#ffcc00", anchor="ne", font=canvas_font)

class LICApp:
    def __init__(self, root):
        self.root = root
        self.root.title("LIC Model Visualizer - Desktop")
        
        # Dynamic Scaling Setup
        self.scale_factor = self.calculate_scale_factor()
        self.setup_geometry()
        
        # Scaled Font Definitions
        self.F_BASE = ("sans-serif", int(14 * self.scale_factor))
        self.F_HEAD = ("sans-serif", int(18 * self.scale_factor), "bold")
        self.F_BTN  = ("sans-serif", int(14 * self.scale_factor), "bold")
        self.F_RUN  = ("sans-serif", int(20 * self.scale_factor), "bold")
        self.F_LOG  = ("monospace", int(12 * self.scale_factor))
        
        self.apply_styles()
        
        self.registry = self.load_interfaces(os.path.join(ROOT_DIR, "Interfaces", "Testing-Interfaces"))
        self.model_configs = {}
        self.log_queue = queue.Queue()
        self.selected_model_names = []
        self.metrics_data = {} # {model_name: {averages: {}, per_image: []}}
        
        self.setup_ui()
        self.poll_log_queue()

    def calculate_scale_factor(self):
        """Calculate scaling factor with more aggressive heuristics for 4K."""
        try:
            # Try to get screen width to guess scaling if DPI detection fails
            sw = self.root.winfo_screenwidth()
            
            # Manual override via env var if needed (e.g. GUI_SCALE=2.0)
            env_scale = os.environ.get("GUI_SCALE")
            if env_scale:
                return float(env_scale)

            # Heuristic: If resolution is 4k (approx 3840 wide)
            if sw >= 3000:
                factor = 2.5
            elif sw >= 2500:
                factor = 2.0
            elif sw >= 1900:
                factor = 1.25
            else:
                factor = 1.0

            # Also check DPI as a fallback/supplement
            dpi = self.root.winfo_fpixels('1i')
            dpi_factor = dpi / 96.0
            
            final_factor = max(factor, dpi_factor)
            
            # Set Tk's internal scaling
            self.root.tk.call('tk', 'scaling', (96 * final_factor) / 72.0)
            
            return final_factor
        except Exception:
            return 1.5 if getattr(self, 'is_high_res', False) else 1.0

    def setup_geometry(self):
        """Set window size based on screen resolution."""
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        
        # Use more screen real estate on high-res displays
        width = int(screen_w * 0.9)
        height = int(screen_h * 0.9)
        
        # Clamp to reasonable values
        width = max(1200, min(width, int(2200 * self.scale_factor)))
        height = max(800, min(height, int(1400 * self.scale_factor)))
        
        self.root.geometry(f"{width}x{height}")

    def apply_styles(self):
        style = ttk.Style()
        if 'clam' in style.theme_names():
            style.theme_use('clam')

        # Increased base font sizes for 4K
        self.F_BASE = ("sans-serif", int(15 * self.scale_factor))
        self.F_HEAD = ("sans-serif", int(20 * self.scale_factor), "bold")
        self.F_BTN  = ("sans-serif", int(15 * self.scale_factor), "bold")
        self.F_RUN  = ("sans-serif", int(24 * self.scale_factor), "bold")
        self.F_LOG  = ("monospace", int(13 * self.scale_factor))

        p_small = int(6 * self.scale_factor)
        p_med = int(12 * self.scale_factor)
        p_large = int(25 * self.scale_factor)

        style.configure('.', font=self.F_BASE)
        style.configure('TLabel', font=self.F_BASE)
        style.configure('Header.TLabel', font=self.F_HEAD, foreground="#003366")

        style.configure('TButton', font=self.F_BTN, padding=p_med)

        style.configure('Run.TButton', font=self.F_RUN, background='#28a745', foreground='white', padding=p_large)
        style.map('Run.TButton', background=[('active', '#218838')])

        style.configure('TLabelframe.Label', font=self.F_HEAD, foreground="#0055a4")

        style.configure('TCheckbutton', font=self.F_BASE)
        style.configure('TCombobox', font=self.F_BASE)

        style.configure('TEntry', font=self.F_BASE, padding=p_small, fieldbackground='white')
        
        style.configure('Vertical.TScrollbar', width=int(20 * self.scale_factor))

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

        sidebar_width = int(550 * self.scale_factor)
        sidebar_padding = int(25 * self.scale_factor)
        self.sidebar = ttk.Frame(self.paned, width=sidebar_width, padding=sidebar_padding)
        self.paned.add(self.sidebar, weight=1)

        ttk.Label(self.sidebar, text="1. Global Settings", style='Header.TLabel').pack(anchor="w", pady=(0, int(15 * self.scale_factor)))

        self.gt_dir_var = tk.StringVar()
        ttk.Label(self.sidebar, text="GT Images Directory:", font=self.F_BASE).pack(anchor="w")
        gt_frame = ttk.Frame(self.sidebar)
        gt_frame.pack(fill=tk.X, pady=(0, int(20 * self.scale_factor)))
        ttk.Entry(gt_frame, textvariable=self.gt_dir_var, font=self.F_BASE).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(gt_frame, text="Browse", width=int(8), command=lambda: self.browse_dir(self.gt_dir_var, check_images=True)).pack(side=tk.LEFT, padx=(int(5 * self.scale_factor),0))

        self.out_dir_var = tk.StringVar(value=os.path.join(ROOT_DIR, "GUI-Visualizer/outputs"))
        ttk.Label(self.sidebar, text="Output Directory:", font=self.F_BASE).pack(anchor="w")
        out_frame = ttk.Frame(self.sidebar)
        out_frame.pack(fill=tk.X, pady=(0, int(20 * self.scale_factor)))
        ttk.Entry(out_frame, textvariable=self.out_dir_var, font=self.F_BASE).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(out_frame, text="Browse", width=int(8), command=lambda: self.browse_dir(self.out_dir_var)).pack(side=tk.LEFT, padx=(int(5 * self.scale_factor),0))

        # Add Evaluation Environment path field
        default_eval_env = os.path.join(ROOT_DIR, "envs/eval-env")
        self.eval_env_var = tk.StringVar(value=default_eval_env if os.path.exists(default_eval_env) else "")
        ttk.Label(self.sidebar, text="Evaluation Env Path (eval-env):", font=self.F_BASE).pack(anchor="w")
        eval_env_frame = ttk.Frame(self.sidebar)
        eval_env_frame.pack(fill=tk.X, pady=(0, int(30 * self.scale_factor)))
        ttk.Entry(eval_env_frame, textvariable=self.eval_env_var, font=self.F_BASE).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(eval_env_frame, text="Browse", width=int(8), command=lambda: self.browse_dir(self.eval_env_var)).pack(side=tk.LEFT, padx=(int(5 * self.scale_factor),0))

        ttk.Label(self.sidebar, text="2. Models", style='Header.TLabel').pack(anchor="w", pady=(0, int(15 * self.scale_factor)))

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
        self.model_listbox.pack(fill=tk.X, pady=(0, int(20 * self.scale_factor)))
        self.model_listbox.bind("<<ListboxSelect>>", self.on_model_select)

        self.run_btn = ttk.Button(self.sidebar, text="RUN EVALUATION", style='Run.TButton', command=self.run_evaluation)
        self.run_btn.pack(fill=tk.X, pady=int(20 * self.scale_factor))

        self.progress = ttk.Progressbar(self.sidebar, mode='indeterminate')
        self.progress.pack(fill=tk.X)

        self.main_area = ttk.Notebook(self.paned)
        self.paned.add(self.main_area, weight=4)

        self.config_tab = ttk.Frame(self.main_area, padding=sidebar_padding)
        self.main_area.add(self.config_tab, text="Configuration")

        self.config_canvas = tk.Canvas(self.config_tab, highlightthickness=0)
        self.config_scrollbar = ttk.Scrollbar(self.config_tab, orient="vertical", command=self.config_canvas.yview)
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

        self.compare_tab = ttk.Frame(self.main_area, padding=int(20 * self.scale_factor))
        self.main_area.add(self.compare_tab, text="Visual Comparison")

        comp_controls = ttk.Frame(self.compare_tab)
        comp_controls.pack(fill=tk.X, pady=(0, int(20 * self.scale_factor)))

        ttk.Label(comp_controls, text="Select Image:", font=self.F_BTN).pack(side=tk.LEFT)
        self.img_selector = ttk.Combobox(comp_controls, state="readonly", width=int(30 * self.scale_factor), font=self.F_BASE)
        self.img_selector.pack(side=tk.LEFT, padx=(int(10 * self.scale_factor), int(30 * self.scale_factor)))
        self.img_selector.bind("<<ComboboxSelected>>", self.update_comparison)

        ttk.Label(comp_controls, text="Select Model:", font=self.F_BTN).pack(side=tk.LEFT)
        self.model_selector = ttk.Combobox(comp_controls, state="readonly", width=int(25 * self.scale_factor), font=self.F_BASE)
        self.model_selector.pack(side=tk.LEFT, padx=int(10 * self.scale_factor))
        self.model_selector.bind("<<ComboboxSelected>>", self.update_comparison)

        self.metrics_btn = ttk.Button(comp_controls, text="Inspect Metrics", command=self.show_current_metrics_popup)
        self.metrics_btn.pack(side=tk.LEFT, padx=int(20 * self.scale_factor))

        self.comp_canvas = ComparisonCanvas(self.compare_tab, bg="#1e1e1e", highlightthickness=0)
        self.comp_canvas.pack(fill=tk.BOTH, expand=True)

        self.metrics_tab = ttk.Frame(self.main_area, padding=sidebar_padding)
        self.main_area.add(self.metrics_tab, text="Metrics Report")

        self.setup_metrics_ui()

        self.log_area = tk.Text(self.sidebar, height=10, font=self.F_LOG, bg="#ffffff", fg="#333333", highlightthickness=1, highlightbackground="#cccccc")
        self.log_area.pack(fill=tk.BOTH, expand=True, pady=(int(25 * self.scale_factor), 0))
        self.log_area.bind("<Key>", self.block_input)


    def setup_metrics_ui(self):
        self.metrics_top = ttk.Frame(self.metrics_tab)
        self.metrics_top.pack(fill=tk.X, pady=(0, int(20 * self.scale_factor)))

        ttk.Label(self.metrics_top, text="Model Performance Summary", style='Header.TLabel').pack(side=tk.LEFT)

        self.metrics_model_sel = ttk.Combobox(self.metrics_top, state="readonly", font=self.F_BASE)
        self.metrics_model_sel.pack(side=tk.RIGHT, padx=int(10 * self.scale_factor))
        self.metrics_model_sel.bind("<<ComboboxSelected>>", self.refresh_metrics_display)
        ttk.Label(self.metrics_top, text="View Model:", font=self.F_BASE).pack(side=tk.RIGHT)

        self.summary_frame = ttk.LabelFrame(self.metrics_tab, text="Averages", padding=int(15 * self.scale_factor))
        self.summary_frame.pack(fill=tk.X, pady=(0, int(20 * self.scale_factor)))
        self.summary_label = ttk.Label(self.summary_frame, text="No evaluation data loaded.", font=self.F_BTN)
        self.summary_label.pack()

        table_frame = ttk.Frame(self.metrics_tab)
        table_frame.pack(fill=tk.BOTH, expand=True)

        columns = ("image", "psnr", "ssim", "lpips", "bpp")
        self.metrics_tree = ttk.Treeview(table_frame, columns=columns, show="headings")

        for col in columns:
            self.metrics_tree.heading(col, text=col.upper())
            self.metrics_tree.column(col, anchor="center", width=int(150 * self.scale_factor))

        scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=self.metrics_tree.yview)
        self.metrics_tree.configure(yscrollcommand=scrollbar.set)

        self.metrics_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

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

    def browse_file(self, var):
        current = var.get()
        initial = os.path.dirname(current) if current and os.path.exists(os.path.dirname(current)) else ROOT_DIR
        path = filedialog.askopenfilename(initialdir=initial)
        if path:
            var.set(os.path.abspath(os.path.expanduser(path)))
    def log(self, msg):
        self.log_queue.put(msg)

    def on_model_select(self, event):
        for widget in self.config_scrollable_frame.winfo_children():
            widget.destroy()

        selected_indices = self.model_listbox.curselection()
        self.selected_model_names = [self.model_listbox.get(i) for i in selected_indices]

        self.model_selector['values'] = self.selected_model_names
        self.metrics_model_sel['values'] = self.selected_model_names

        for name in self.selected_model_names:
            self.build_model_config_ui(name)

    def build_model_config_ui(self, model_name):
        frame = ttk.LabelFrame(self.config_scrollable_frame, text=f"Config: {model_name}", padding=int(20 * self.scale_factor))
        frame.pack(fill=tk.X, pady=int(15 * self.scale_factor), padx=int(10 * self.scale_factor))

        interface_cls = self.registry[model_name]
        required_args = getattr(interface_cls, 'REQUIRED_ARGS', [])

        if model_name not in self.model_configs:
            self.model_configs[model_name] = {
                "args": {},
                "workdir": tk.StringVar(value=f"LIC-Models/{model_name}"),
                "env": tk.StringVar()
            }

            defaults = getattr(interface_cls, 'DEFAULT_VARS', {})
            for k, v in defaults.items():
                if k in ['data', 'dataset', 'input', 'input_dir']: continue
                self.model_configs[model_name]["args"][k] = tk.StringVar(value=str(v))

            for req in required_args:
                if req in ['data', 'dataset', 'input', 'input_dir']: continue
                if req not in self.model_configs[model_name]["args"]:
                    self.model_configs[model_name]["args"][req] = tk.StringVar()

        config = self.model_configs[model_name]

        def create_row(label_text, var, is_dir=False, is_file=False, required=False):
            row = ttk.Frame(frame)
            row.pack(fill=tk.X, pady=int(8 * self.scale_factor))

            display_text = label_text
            if required:
                display_text = "* " + display_text

            lbl = ttk.Label(row, text=display_text, width=int(25), font=self.F_BTN)
            lbl.pack(side=tk.LEFT)
            if required:
                lbl.configure(foreground="#cc0000")

            entry = ttk.Entry(row, textvariable=var, font=self.F_BASE)
            entry.pack(side=tk.LEFT, fill=tk.X, expand=True)

            if is_dir or is_file:
                ttk.Button(row, text="Browse", width=int(8), command=lambda: self.browse_dir(var) if is_dir else self.browse_file(var)).pack(side=tk.LEFT, padx=(int(5 * self.scale_factor),0))

        create_row("Working Dir:", config["workdir"], is_dir=True, required=True)
        create_row("Env Path:", config["env"], is_dir=True)

        ttk.Separator(frame, orient='horizontal').pack(fill=tk.X, pady=int(15 * self.scale_factor))

        for arg_name, var in config["args"].items():
            arg_lower = arg_name.lower()
            
            # Heuristic to distinguish between file and directory inputs
            is_file = any(x in arg_lower for x in ['checkpoint', 'model', 'file', 'elic', 'codec', 'pth', 'pkl', 'weights'])
            is_dir = any(x in arg_lower for x in ['dir', 'dataset', 'folder', 'save'])
            
            # Ambiguous 'path' defaults to directory unless it looks like a known file type above
            if 'path' in arg_lower and not is_file and not is_dir:
                is_dir = True
                
            is_required = arg_name in required_args
            
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
        # 0. Check for Evaluation Environment
        eval_env_path = self.eval_env_var.get().strip()
        if not eval_env_path or not os.path.exists(eval_env_path):
            target_path = eval_env_path if eval_env_path else os.path.join(ROOT_DIR, "envs/eval-env")
            if messagebox.askyesno("Missing Evaluation Environment", 
                                   f"The evaluation environment was not found at:\n{target_path}\n\n"
                                   "Would you like to install it now? (Requires Conda)"):
                self.log(f"[GUI] Starting installation of evaluation environment to: {target_path}\n")
                try:
                    # Dynamically load create-env.py logic
                    spec = importlib.util.spec_from_file_location("create_env", os.path.join(ROOT_DIR, "create-env.py"))
                    ce = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(ce)
                    
                    req_path = os.path.join(ROOT_DIR, "evaluation-requirements.txt")
                    
                    # Redirect stdout for the setup function (it uses print)
                    orig_print = builtins.print
                    def thread_print(*args, **kwargs):
                        msg = " ".join(map(str, args))
                        end = kwargs.get('end', '\n')
                        self.log(msg + end)
                    builtins.print = thread_print
                    
                    ce.setup_conda_env(target_path, req_path, "3.10")
                    
                    builtins.print = orig_print
                    self.eval_env_var.set(target_path)
                    self.log(f"[GUI] Evaluation environment installed successfully.\n")
                except Exception as e:
                    self.log(f"[GUI ERROR] Failed to install evaluation environment: {e}\n")
                    if not messagebox.askyesno("Installation Failed", "Evaluation environment setup failed. Continue anyway?"):
                        self.root.after(0, self.finish_execution)
                        return
            else:
                self.log("[GUI] Skipping evaluation environment installation. Evaluation step may use system Python.\n")

        output_base = os.path.expanduser(self.out_dir_var.get().strip())
        
        # 1. Prepare Dispatcher config
        tasks = {}
        eval_tasks = []
        
        for model_name in self.selected_model_names:
            config = self.model_configs[model_name]
            final_args = {k: v.get() for k, v in config["args"].items()}
            
            # Map input
            interface_cls = self.registry[model_name]
            data_keys = ['input', 'input_dir', 'data', 'dataset']
            target_key = next((k for k in data_keys if k in interface_cls.REQUIRED_ARGS or k in getattr(interface_cls, 'DEFAULT_VARS', {})), "dataset")
            final_args[target_key] = gt_dir
            
            model_out = os.path.abspath(os.path.join(output_base, model_name))
            os.makedirs(model_out, exist_ok=True)
            
            # Interface handles save_dir internal splitting
            final_args["save_dir"] = model_out

            tasks[model_name] = {
                "task_name": model_name,
                "directory": os.path.join(ROOT_DIR, config["workdir"].get()),
                "env_path": config["env"].get() or None,
                "arguments": final_args
            }
            
            eval_tasks.append({
                "task_name": model_name,
                "save_dir": model_out,
                "input_dir": gt_dir
            })

        eval_env_path = self.eval_env_var.get().strip()
        gui_config = {
            "testing": {
                "tasks": tasks,
                "evaluation": {
                    "env_path": eval_env_path if eval_env_path and os.path.exists(eval_env_path) else "n/a",
                    "tasks": eval_tasks
                }
            }
        }
        
        config_path = os.path.join(ROOT_DIR, "gui_arguments.json")
        with open(config_path, 'w') as f:
            json.dump(gui_config, f, indent=4)

        # 2. Setup Log Redirection & Monkey-patching
        original_print = builtins.print
        def custom_print(*args, **kwargs):
            msg = " ".join(map(str, args))
            end = kwargs.get('end', '\n')
            self.log(msg + end)
        builtins.print = custom_print

        original_input = builtins.input
        def custom_input(prompt=""):
            self.log(f"{prompt} [Waiting for user popup...]\n")
            # Show a yes/no dialog. Tkinter messageboxes are generally thread-safe 
            # for display, but they block the calling thread (execution_thread) 
            # until dismissed, which is the desired behavior for input().
            response_bool = messagebox.askyesno("Dependency / Confirmation Required", prompt, parent=self.root)
            response = "y" if response_bool else "n"
            self.log(f"User selected: {response}\n")
            return response
        builtins.input = custom_input

        # Monkey-patch subprocess to capture model output
        orig_run = subprocess.run
        orig_check_call = subprocess.check_call

        def streaming_executor(cmd, **kwargs):
            # Ensure output is captured
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

        # 3. Execute via Dispatcher
        try:
            dispatcher = Dispatcher(
                arg_json_path=config_path,
                run_test=True,
                test_interfaces_path=os.path.join(ROOT_DIR, "Interfaces", "Testing-Interfaces")
            )
            dispatcher.run()
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
        
        for model_name in self.selected_model_names:
            metrics_file = os.path.join(output_base, model_name, f"{model_name}_metrics.json")
            if os.path.exists(metrics_file):
                try:
                    with open(metrics_file, 'r') as f:
                        self.metrics_data[model_name] = json.load(f)
                except Exception as e:
                    self.log(f"[ERROR] Failed to load metrics for {model_name}: {e}\n")
        
        if self.metrics_data:
            first_model = list(self.metrics_data.keys())[0]
            self.metrics_model_sel.set(first_model)
            self.refresh_metrics_display()

    def refresh_metrics_display(self, event=None):
        model_name = self.metrics_model_sel.get()
        if not model_name or model_name not in self.metrics_data:
            return
            
        data = self.metrics_data[model_name]
        
        # Update Summary
        avg = data.get("averages", {})
        summary_text = f"PSNR: {avg.get('psnr', 'N/A')} dB | SSIM: {avg.get('ssim', 'N/A')} | LPIPS: {avg.get('lpips', 'N/A')} | BPP: {avg.get('bpp', 'N/A')}"
        self.summary_label.config(text=summary_text)
        
        # Update Table
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

    def refresh_image_list(self):
        gt_dir = os.path.expanduser(self.gt_dir_var.get().strip())
        if os.path.isdir(gt_dir):
            imgs = sorted([os.path.basename(f) for f in glob.glob(os.path.join(gt_dir, "*")) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            self.img_selector['values'] = imgs
            if imgs: self.img_selector.current(0)

    def update_comparison(self, event=None):
        selected_img = self.img_selector.get()
        model_name = self.model_selector.get()
        
        gt_dir = os.path.expanduser(self.gt_dir_var.get().strip())
        out_base = os.path.expanduser(self.out_dir_var.get().strip())
        
        if not selected_img or not model_name: return
        
        gt_path = os.path.join(gt_dir, selected_img)
        model_out = os.path.join(out_base, model_name, "reconstruction")
        if not os.path.exists(model_out):
            model_out = os.path.join(out_base, model_name) # Fallback
        
        base_name = os.path.splitext(selected_img)[0]
        
        found = None
        if os.path.exists(model_out):
            for f in os.listdir(model_out):
                if f.startswith(base_name) and f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    found = os.path.join(model_out, f)
                    break
                # Also try without prefix if needed, or exact match
                if f == selected_img:
                    found = os.path.join(model_out, f)
                    break

        if found:
            self.comp_canvas.set_images(gt_path, found)
            self.log(f"[Viewer] Loaded: {os.path.basename(found)}\n")
        else:
            self.comp_canvas.set_images(gt_path, None) 
            self.log(f"[Viewer] Warning: No recon found for {base_name} in {model_name}.\n")

    def show_current_metrics_popup(self):
        selected_img = self.img_selector.get()
        model_name = self.model_selector.get()
        
        if not selected_img or not model_name:
            messagebox.showwarning("Warning", "Please select an image and model first.")
            return
            
        if model_name not in self.metrics_data:
            messagebox.showinfo("Info", "No metrics available. Please run evaluation first.")
            return
            
        data = self.metrics_data[model_name]
        img_metrics = None
        for item in data.get("per_image_metrics", []):
            if item.get("image_name") == selected_img or item.get("image_name").startswith(os.path.splitext(selected_img)[0]):
                img_metrics = item
                break
        
        popup = tk.Toplevel(self.root)
        popup.title(f"Metrics: {selected_img} ({model_name})")
        
        # Scaled popup size
        pw = int(600 * self.scale_factor)
        ph = int(500 * self.scale_factor)
        popup.geometry(f"{pw}x{ph}")
        popup.transient(self.root)
        
        frame = ttk.Frame(popup, padding=int(30 * self.scale_factor))
        frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(frame, text=f"Image: {selected_img}", style='Header.TLabel').pack(pady=(0, int(20 * self.scale_factor)))
        
        avg = data.get("averages", {})
        
        def add_metric(name, value, avg_val):
            row = ttk.Frame(frame)
            row.pack(fill=tk.X, pady=int(10 * self.scale_factor))
            ttk.Label(row, text=f"{name}:", font=self.F_BTN, width=int(12)).pack(side=tk.LEFT)
            ttk.Label(row, text=str(value), font=self.F_BTN, foreground="#d9534f").pack(side=tk.LEFT)
            ttk.Label(row, text=f" (Model Avg: {avg_val})", font=self.F_BASE).pack(side=tk.LEFT)

        if img_metrics:
            add_metric("PSNR", img_metrics.get("psnr"), avg.get("psnr"))
            add_metric("SSIM", img_metrics.get("ssim"), avg.get("ssim"))
            add_metric("LPIPS", img_metrics.get("lpips"), avg.get("lpips"))
            add_metric("BPP", img_metrics.get("bpp"), avg.get("bpp"))
        else:
            ttk.Label(frame, text="Specific metrics for this image not found in the report.", wraplength=int(400 * self.scale_factor)).pack()

        ttk.Button(frame, text="Close", command=popup.destroy).pack(pady=int(20 * self.scale_factor))

if __name__ == "__main__":
    root = tk.Tk()
    app = LICApp(root)
    root.mainloop()