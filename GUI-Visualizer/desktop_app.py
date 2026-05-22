import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import sys
import threading
import importlib.util
import inspect
from PIL import Image, ImageTk
import glob
import builtins
import queue

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
        
        canvas_font = ("sans-serif", 28, "bold")
        
        if self.scaled_img2:
            line_x = offset_x + split_x
            self.create_line(line_x, offset_y, line_x, offset_y + self.new_h, fill="#00ffff", width=5)
            self.create_text(offset_x + 20, offset_y + 20, text="COMPRESSED", fill="#00ffff", anchor="nw", font=canvas_font)
            self.create_text(offset_x + self.new_w - 20, offset_y + 20, text="GROUND TRUTH", fill="#00ffff", anchor="ne", font=canvas_font)
        else:
            self.create_text(offset_x + self.new_w - 20, offset_y + 20, text="GROUND TRUTH (Awaiting Output...)", fill="#ffcc00", anchor="ne", font=canvas_font)

class LICApp:
    def __init__(self, root):
        self.root = root
        self.root.title("LIC Model Visualizer - Desktop")
        self.root.geometry("1800x1100") 
        
        self.F_BASE = ("sans-serif", 16)
        self.F_HEAD = ("sans-serif", 22, "bold")
        self.F_BTN  = ("sans-serif", 16, "bold")
        self.F_RUN  = ("sans-serif", 24, "bold")
        self.F_LOG  = ("monospace", 14)
        
        self.apply_styles()
        
        self.registry = self.load_interfaces(os.path.join(ROOT_DIR, "Interfaces", "Testing-Interfaces"))
        self.model_configs = {}
        self.log_queue = queue.Queue()
        self.selected_model_names = []
        
        self.setup_ui()
        self.poll_log_queue()

    def apply_styles(self):
        style = ttk.Style()
        if 'clam' in style.theme_names():
            style.theme_use('clam')
            
        style.configure('.', font=self.F_BASE)
        style.configure('TLabel', font=self.F_BASE)
        style.configure('Header.TLabel', font=self.F_HEAD, foreground="#003366")
        
        style.configure('TButton', font=self.F_BTN, padding=10)
        
        style.configure('Run.TButton', font=self.F_RUN, background='#28a745', foreground='white', padding=20)
        style.map('Run.TButton', background=[('active', '#218838')])
        
        style.configure('TLabelframe.Label', font=self.F_HEAD, foreground="#0055a4")
        
        style.configure('TCheckbutton', font=self.F_BASE)
        style.configure('TCombobox', font=self.F_BASE)

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

        self.sidebar = ttk.Frame(self.paned, width=550, padding=25)
        self.paned.add(self.sidebar, weight=1)

        ttk.Label(self.sidebar, text="1. Global Settings", style='Header.TLabel').pack(anchor="w", pady=(0, 15))
        
        self.gt_dir_var = tk.StringVar()
        ttk.Label(self.sidebar, text="GT Images Directory:", font=self.F_BASE).pack(anchor="w")
        gt_frame = ttk.Frame(self.sidebar)
        gt_frame.pack(fill=tk.X, pady=(0, 20))
        ttk.Entry(gt_frame, textvariable=self.gt_dir_var, font=self.F_BASE).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(gt_frame, text="...", width=4, command=lambda: self.browse_dir(self.gt_dir_var)).pack(side=tk.LEFT, padx=(5,0))

        self.out_dir_var = tk.StringVar(value=os.path.join(ROOT_DIR, "GUI-Visualizer/outputs"))
        ttk.Label(self.sidebar, text="Output Directory:", font=self.F_BASE).pack(anchor="w")
        out_frame = ttk.Frame(self.sidebar)
        out_frame.pack(fill=tk.X, pady=(0, 30))
        ttk.Entry(out_frame, textvariable=self.out_dir_var, font=self.F_BASE).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(out_frame, text="...", width=4, command=lambda: self.browse_dir(self.out_dir_var)).pack(side=tk.LEFT, padx=(5,0))

        ttk.Label(self.sidebar, text="2. Models", style='Header.TLabel').pack(anchor="w", pady=(0, 15))
        
        self.model_listbox = tk.Listbox(
            self.sidebar, 
            selectmode=tk.MULTIPLE, 
            height=6, 
            exportselection=False,
            font=self.F_BASE, 
            bg="#2b2b2b",       
            fg="#ffffff",       
            selectbackground="#007acc", 
            selectforeground="#ffffff"
        )
        for name in sorted(self.registry.keys()):
            self.model_listbox.insert(tk.END, name)
        self.model_listbox.pack(fill=tk.X, pady=(0, 20))
        self.model_listbox.bind("<<ListboxSelect>>", self.on_model_select)

        self.auto_confirm_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.sidebar, text="Auto-confirm dependency installs", variable=self.auto_confirm_var).pack(anchor="w", pady=(0, 20))

        self.run_btn = ttk.Button(self.sidebar, text="RUN EVALUATION", style='Run.TButton', command=self.run_evaluation)
        self.run_btn.pack(fill=tk.X, pady=20)

        self.progress = ttk.Progressbar(self.sidebar, mode='indeterminate')
        self.progress.pack(fill=tk.X)

        self.main_area = ttk.Notebook(self.paned)
        self.paned.add(self.main_area, weight=4)

        self.config_tab = ttk.Frame(self.main_area, padding=25)
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

        self.config_canvas.pack(side="left", fill="both", expand=True)
        self.config_scrollbar.pack(side="right", fill="y")

        self.compare_tab = ttk.Frame(self.main_area, padding=20)
        self.main_area.add(self.compare_tab, text="Visual Comparison")
        
        comp_controls = ttk.Frame(self.compare_tab)
        comp_controls.pack(fill=tk.X, pady=(0, 20))
        
        ttk.Label(comp_controls, text="Select Image:", font=self.F_BTN).pack(side=tk.LEFT)
        self.img_selector = ttk.Combobox(comp_controls, state="readonly", width=30, font=self.F_BASE)
        self.img_selector.pack(side=tk.LEFT, padx=(10, 30))
        self.img_selector.bind("<<ComboboxSelected>>", self.update_comparison)
        
        ttk.Label(comp_controls, text="Select Model:", font=self.F_BTN).pack(side=tk.LEFT)
        self.model_selector = ttk.Combobox(comp_controls, state="readonly", width=25, font=self.F_BASE)
        self.model_selector.pack(side=tk.LEFT, padx=10)
        self.model_selector.bind("<<ComboboxSelected>>", self.update_comparison)

        self.comp_canvas = ComparisonCanvas(self.compare_tab, bg="#1e1e1e", highlightthickness=0)
        self.comp_canvas.pack(fill=tk.BOTH, expand=True)

        self.log_area = tk.Text(self.sidebar, height=10, font=self.F_LOG, bg="#f4f4f4", fg="#333333")
        self.log_area.pack(fill=tk.BOTH, expand=True, pady=(25, 0))
        self.log_area.bind("<Key>", self.block_input)

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

    def browse_dir(self, var):
        path = filedialog.askdirectory()
        if path:
            path = os.path.expanduser(path)
            try:
                has_images = any(f.lower().endswith(('.png', '.jpg', '.jpeg')) for f in os.listdir(path))
                if not has_images:
                    messagebox.showwarning("Warning", "The selected directory appears to have no images.")
            except Exception as e:
                self.log(f"[ERROR] Could not read directory {path}: {e}\n")

            var.set(path)
            self.refresh_image_list()
            
    def browse_file(self, var):
        path = filedialog.askopenfilename()
        if path:
            var.set(os.path.expanduser(path))

    def log(self, msg):
        self.log_queue.put(msg)

    def on_model_select(self, event):
        for widget in self.config_scrollable_frame.winfo_children():
            widget.destroy()

        selected_indices = self.model_listbox.curselection()
        self.selected_model_names = [self.model_listbox.get(i) for i in selected_indices]
        
        self.model_selector['values'] = self.selected_model_names
        
        for name in self.selected_model_names:
            self.build_model_config_ui(name)

    def build_model_config_ui(self, model_name):
        frame = ttk.LabelFrame(self.config_scrollable_frame, text=f"Config: {model_name}", padding=20)
        frame.pack(fill=tk.X, pady=15, padx=10)
        
        interface_cls = self.registry[model_name]
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
            
            for req in getattr(interface_cls, 'REQUIRED_ARGS', []):
                if req in ['data', 'dataset', 'input', 'input_dir']: continue
                if req not in self.model_configs[model_name]["args"]:
                    self.model_configs[model_name]["args"][req] = tk.StringVar()

        config = self.model_configs[model_name]
        
        def create_row(label_text, var, is_dir=False, is_file=False):
            row = ttk.Frame(frame)
            row.pack(fill=tk.X, pady=8) 
            ttk.Label(row, text=label_text, width=20, font=self.F_BTN).pack(side=tk.LEFT)
            ttk.Entry(row, textvariable=var, font=self.F_BASE).pack(side=tk.LEFT, fill=tk.X, expand=True)
            
            if is_dir:
                ttk.Button(row, text="...", width=4, command=lambda: self.browse_dir(var)).pack(side=tk.LEFT, padx=(5,0))
            elif is_file:
                ttk.Button(row, text="...", width=4, command=lambda: self.browse_file(var)).pack(side=tk.LEFT, padx=(5,0))

        create_row("Working Dir:", config["workdir"], is_dir=True)
        create_row("Env Path:", config["env"], is_dir=True)

        ttk.Separator(frame, orient='horizontal').pack(fill=tk.X, pady=15)

        for arg_name, var in config["args"].items():
            arg_lower = arg_name.lower()
            is_file = any(x in arg_lower for x in ['checkpoint', 'model', 'file'])
            is_dir = any(x in arg_lower for x in ['dir', 'dataset', 'folder', 'path', 'save']) and not is_file
            
            create_row(f"{arg_name}:", var, is_dir=is_dir, is_file=is_file)

    def run_evaluation(self):
        if not self.selected_model_names:
            messagebox.showerror("Error", "Please select at least one model from the Models list.")
            return

        gt_dir = os.path.expanduser(self.gt_dir_var.get().strip())
        if not gt_dir or not os.path.isdir(gt_dir):
            messagebox.showerror("Error", f"Valid GT directory required.\nCurrent path: {gt_dir}")
            return
            
        images = glob.glob(os.path.join(gt_dir, "*"))
        image_files = [f for f in images if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not image_files:
            messagebox.showerror("Error", "The selected directory does not contain any images (.png, .jpg, .jpeg).")
            return

        self.run_btn.config(state=tk.DISABLED)
        self.progress.start()
        
        threading.Thread(target=self.execution_thread, args=(gt_dir,), daemon=True).start()

    def execution_thread(self, gt_dir):
        output_base = os.path.expanduser(self.out_dir_var.get().strip())
        
        for model_name in self.selected_model_names:
            self.log(f"--- Starting {model_name} ---\n")
            config = self.model_configs[model_name]
            interface_cls = self.registry[model_name]
            
            final_args = {k: v.get() for k, v in config["args"].items()}
            
            data_keys = ['input', 'input_dir', 'data', 'dataset']
            target_key = next((k for k in data_keys if k in interface_cls.REQUIRED_ARGS or k in getattr(interface_cls, 'DEFAULT_VARS', {})), "dataset")
            final_args[target_key] = gt_dir
            
            model_out = os.path.join(output_base, model_name)
            os.makedirs(model_out, exist_ok=True)
            if model_name == "LIC-HPCM": final_args["save_dir"] = model_out
            elif model_name == "StableCodec":
                final_args["rec_path"] = model_out
                final_args["bin_path"] = os.path.join(model_out, "bins")
            elif model_name == "DCVC-RT": final_args["save_dir"] = model_out
            elif model_name == "ELIC": final_args["experiment"] = f"gui_eval_{model_name}"
            elif model_name == "RwkvCompress": final_args["output_dir"] = model_out

            try:
                import subprocess as sp_module
                orig_run = sp_module.run
                orig_check_call = sp_module.check_call

                def streaming_executor(cmd, **kwargs):
                    kwargs.pop('check', None)
                    kwargs.pop('capture_output', None)
                    kwargs['stdout'] = sp_module.PIPE
                    kwargs['stderr'] = sp_module.STDOUT
                    kwargs['text'] = True
                    kwargs['bufsize'] = 1
                    kwargs['universal_newlines'] = True
                    
                    process = sp_module.Popen(cmd, **kwargs)
                    if process.stdout:
                        for line in process.stdout:
                            self.log(line)
                    return process.wait()

                def custom_check_call(cmd, *args, **kwargs):
                    ret = streaming_executor(cmd, **kwargs)
                    if ret != 0: raise sp_module.CalledProcessError(ret, cmd)
                    return 0

                def custom_run(cmd, *args, **kwargs):
                    ret = streaming_executor(cmd, **kwargs)
                    return sp_module.CompletedProcess(cmd, ret)

                sp_module.run = custom_run
                sp_module.check_call = custom_check_call

                original_print = builtins.print
                def custom_print(*args, **kwargs):
                    msg = " ".join(map(str, args))
                    end = kwargs.get('end', '\n')
                    self.log(msg + end)
                builtins.print = custom_print
                
                original_input = builtins.input
                auto_confirm = self.auto_confirm_var.get()
                def custom_input(prompt=""):
                    response = "y" if auto_confirm else "n"
                    self.log(f"{prompt} [GUI Auto-respond: '{response}']\n")
                    return response
                builtins.input = custom_input
                
                interface = interface_cls(job_args=final_args)
                interface.WORKING_DIR = os.path.join(ROOT_DIR, config["workdir"].get())
                if config["env"].get():
                    interface.ENV_PATH = config["env"].get()
                
                interface.execute()
                
                sp_module.run = orig_run
                sp_module.check_call = orig_check_call
                builtins.print = original_print
                builtins.input = original_input
                self.log(f"[SUCCESS] {model_name} finished.\n")
            except Exception as e:
                self.log(f"[ERROR] {model_name} failed: {e}\n")
                sp_module.run = orig_run
                sp_module.check_call = orig_check_call
                builtins.print = original_print
                builtins.input = original_input

        self.root.after(0, self.finish_execution)

    def finish_execution(self):
        self.progress.stop()
        self.run_btn.config(state=tk.NORMAL)
        self.refresh_image_list()
        messagebox.showinfo("Done", "Evaluation complete.")

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
        model_out = os.path.join(out_base, model_name)
        
        # Get the name without extension (e.g., "000000000724")
        base_name = os.path.splitext(selected_img)[0]
        
        found = None
        if os.path.exists(model_out):
            # Look for any file in model_out that starts with our base_name
            for f in os.listdir(model_out):
                if f.startswith(base_name) and f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    found = os.path.join(model_out, f)
                    break

        if found:
            self.comp_canvas.set_images(gt_path, found)
            self.log(f"[Viewer] Loaded: {os.path.basename(found)}\n")
        else:
            self.comp_canvas.set_images(gt_path, None) 
            self.log(f"[Viewer] Warning: No recon found for {base_name} in {model_name}.\n")

if __name__ == "__main__":
    root = tk.Tk()
    
    try:
        root.tk.call('tk', 'scaling', 3.0)
    except Exception as e:
        print(f"Scaling failed: {e}")
        
    app = LICApp(root)
    root.mainloop()