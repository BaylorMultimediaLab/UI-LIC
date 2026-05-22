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

# Add project root to path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

class ComparisonCanvas(tk.Canvas):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.image1 = None # GT
        self.image2 = None # Compressed
        self.tk_image1 = None
        self.tk_image2 = None
        self.slider_pos = 0.5
        
        self.bind("<Configure>", self.on_resize)
        self.bind("<B1-Motion>", self.on_drag)
        self.bind("<Button-1>", self.on_drag)

    def set_images(self, path1, path2):
        try:
            self.image1 = Image.open(path1).convert("RGB")
            self.image2 = Image.open(path2).convert("RGB")
            self.render()
        except Exception as e:
            print(f"Error loading images: {e}")

    def on_resize(self, event):
        self.render()

    def on_drag(self, event):
        width = self.winfo_width()
        if width > 0:
            self.slider_pos = max(0, min(1, event.x / width))
            self.render()

    def render(self):
        if not self.image1 or not self.image2:
            return

        canvas_width = self.winfo_width()
        canvas_height = self.winfo_height()
        
        if canvas_width < 10 or canvas_height < 10:
            return

        # Resize images to fit canvas while maintaining aspect ratio
        img_w, img_h = self.image1.size
        ratio = min(canvas_width / img_w, canvas_height / img_h)
        new_w = int(img_w * ratio)
        new_h = int(img_h * ratio)

        res1 = self.image1.resize((new_w, new_h), Image.LANCZOS)
        res2 = self.image2.resize((new_w, new_h), Image.LANCZOS)

        # Create the combined image based on slider_pos
        split_x = int(new_w * self.slider_pos)
        
        combined = Image.new("RGB", (new_w, new_h))
        # Left part from image2 (Compressed) - traditionally people like "after" on left or right?
        # Let's do image1 (GT) on right, image2 (Compressed) on left
        left_part = res2.crop((0, 0, split_x, new_h))
        right_part = res1.crop((split_x, 0, new_w, new_h))
        
        combined.paste(left_part, (0, 0))
        combined.paste(right_part, (split_x, 0))

        self.tk_image = ImageTk.PhotoImage(combined)
        self.delete("all")
        
        # Center image
        offset_x = (canvas_width - new_w) // 2
        offset_y = (canvas_height - new_h) // 2
        
        self.create_image(offset_x, offset_y, anchor="nw", image=self.tk_image)
        
        # Draw slider line
        line_x = offset_x + split_x
        self.create_line(line_x, offset_y, line_x, offset_y + new_h, fill="white", width=2)
        
        # Labels
        self.create_text(offset_x + 10, offset_y + 10, text="COMPRESSED", fill="white", anchor="nw", font=("Arial", 10, "bold"))
        self.create_text(offset_x + new_w - 10, offset_y + 10, text="GROUND TRUTH", fill="white", anchor="ne", font=("Arial", 10, "bold"))

class LICApp:
    def __init__(self, root):
        self.root = root
        self.root.title("LIC Model Visualizer - Desktop")
        self.root.geometry("1200x800")
        
        self.registry = self.load_interfaces(os.path.join(ROOT_DIR, "Interfaces", "Testing-Interfaces"))
        self.model_configs = {}
        self.log_queue = queue.Queue()
        
        self.setup_ui()
        self.poll_log_queue()

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
        # Main Paned Window
        self.paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.paned.pack(fill=tk.BOTH, expand=True)

        # Sidebar
        self.sidebar = ttk.Frame(self.paned, width=300, padding=10)
        self.paned.add(self.sidebar, weight=1)

        ttk.Label(self.sidebar, text="1. Global Settings", font=("Arial", 12, "bold")).pack(anchor="w", pady=(0, 5))
        
        self.gt_dir_var = tk.StringVar()
        ttk.Label(self.sidebar, text="GT Images Directory:").pack(anchor="w")
        gt_frame = ttk.Frame(self.sidebar)
        gt_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Entry(gt_frame, textvariable=self.gt_dir_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(gt_frame, text="...", width=3, command=lambda: self.browse_dir(self.gt_dir_var)).pack(side=tk.LEFT)

        self.out_dir_var = tk.StringVar(value=os.path.join(ROOT_DIR, "GUI-Visualizer/outputs"))
        ttk.Label(self.sidebar, text="Output Directory:").pack(anchor="w")
        out_frame = ttk.Frame(self.sidebar)
        out_frame.pack(fill=tk.X, pady=(0, 20))
        ttk.Entry(out_frame, textvariable=self.out_dir_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(out_frame, text="...", width=3, command=lambda: self.browse_dir(self.out_dir_var)).pack(side=tk.LEFT)

        ttk.Label(self.sidebar, text="2. Models", font=("Arial", 12, "bold")).pack(anchor="w", pady=(0, 5))
        self.model_listbox = tk.Listbox(self.sidebar, selectmode=tk.MULTIPLE, height=10)
        for name in sorted(self.registry.keys()):
            self.model_listbox.insert(tk.END, name)
        self.model_listbox.pack(fill=tk.X, pady=(0, 10))
        self.model_listbox.bind("<<ListboxSelect>>", self.on_model_select)

        self.auto_confirm_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.sidebar, text="Auto-confirm dependency installs", variable=self.auto_confirm_var).pack(anchor="w", pady=(0, 5))

        self.run_btn = ttk.Button(self.sidebar, text="🚀 RUN EVALUATION", command=self.run_evaluation)
        self.run_btn.pack(fill=tk.X, pady=10)

        self.progress = ttk.Progressbar(self.sidebar, mode='indeterminate')
        self.progress.pack(fill=tk.X)

        # Right Area (Tabs)
        self.main_area = ttk.Notebook(self.paned)
        self.paned.add(self.main_area, weight=4)

        # Tab 1: Configuration
        self.config_tab = ttk.Frame(self.main_area, padding=10)
        self.main_area.add(self.config_tab, text="Configuration")
        
        self.config_canvas = tk.Canvas(self.config_tab)
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

        # Tab 2: Comparison
        self.compare_tab = ttk.Frame(self.main_area, padding=10)
        self.main_area.add(self.compare_tab, text="Visual Comparison")
        
        comp_controls = ttk.Frame(self.compare_tab)
        comp_controls.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(comp_controls, text="Select Image:").pack(side=tk.LEFT)
        self.img_selector = ttk.Combobox(comp_controls, state="readonly", width=30)
        self.img_selector.pack(side=tk.LEFT, padx=5)
        self.img_selector.bind("<<ComboboxSelected>>", self.update_comparison)
        
        ttk.Label(comp_controls, text="Select Model:").pack(side=tk.LEFT, padx=(10, 0))
        self.model_selector = ttk.Combobox(comp_controls, state="readonly", width=20)
        self.model_selector.pack(side=tk.LEFT, padx=5)
        self.model_selector.bind("<<ComboboxSelected>>", self.update_comparison)

        self.comp_canvas = ComparisonCanvas(self.compare_tab, bg="black")
        self.comp_canvas.pack(fill=tk.BOTH, expand=True)

        # Log Area
        self.log_area = tk.Text(self.sidebar, height=15, font=("Courier", 9))
        self.log_area.pack(fill=tk.X, pady=(20, 0))
        self.log_area.bind("<Key>", self.block_input)

    def block_input(self, event):
        # Allow Copy (Cmd+C on Mac or Ctrl+C) and Select All
        # State bits: 0x4=Control, 0x8=Mod1/Alt, 0x10=Mod2/Command on some systems
        is_modifier = event.state & (0x4 | 0x8 | 0x10 | 0x40)
        if is_modifier and event.keysym.lower() in ('c', 'a'):
            return None
        # Allow selection/navigation keys
        if event.keysym in ('Left', 'Right', 'Up', 'Down', 'Prior', 'Next', 'Home', 'End'):
            return None
        return "break"

    def poll_log_queue(self):
        """Periodically check for new logs to avoid flooding the UI thread."""
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
            var.set(path)
            self.refresh_image_list()

    def browse_file(self, var):
        path = filedialog.askopenfilename()
        if path:
            var.set(path)

    def log(self, msg):
        self.log_queue.put(msg)

    def on_model_select(self, event):
        # Clear existing configs
        for widget in self.config_scrollable_frame.winfo_children():
            widget.destroy()

        selected_indices = self.model_listbox.curselection()
        self.selected_model_names = [self.model_listbox.get(i) for i in selected_indices]
        
        self.model_selector['values'] = self.selected_model_names
        
        for name in self.selected_model_names:
            self.build_model_config_ui(name)

    def build_model_config_ui(self, model_name):
        frame = ttk.LabelFrame(self.config_scrollable_frame, text=f"Config: {model_name}", padding=10)
        frame.pack(fill=tk.X, pady=5, padx=5)
        
        interface_cls = self.registry[model_name]
        if model_name not in self.model_configs:
            self.model_configs[model_name] = {
                "args": {},
                "workdir": tk.StringVar(value=f"LIC-Models/{model_name}"),
                "env": tk.StringVar()
            }
            
            # Init args with defaults
            defaults = getattr(interface_cls, 'DEFAULT_VARS', {})
            for k, v in defaults.items():
                if k in ['data', 'dataset', 'input', 'input_dir']: continue
                self.model_configs[model_name]["args"][k] = tk.StringVar(value=str(v))
            
            # Ensure required args are present
            for req in getattr(interface_cls, 'REQUIRED_ARGS', []):
                if req in ['data', 'dataset', 'input', 'input_dir']: continue
                if req not in self.model_configs[model_name]["args"]:
                    self.model_configs[model_name]["args"][req] = tk.StringVar()

        config = self.model_configs[model_name]
        
        # Working Dir
        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="Working Dir:", width=15).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=config["workdir"]).pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Env Path
        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="Env Path:", width=15).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=config["env"]).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(row, text="...", width=3, command=lambda: self.browse_dir(config["env"])).pack(side=tk.LEFT)

        # Arguments
        for arg_name, var in config["args"].items():
            row = ttk.Frame(frame)
            row.pack(fill=tk.X, pady=2)
            ttk.Label(row, text=f"{arg_name}:", width=15).pack(side=tk.LEFT)
            ttk.Entry(row, textvariable=var).pack(side=tk.LEFT, fill=tk.X, expand=True)
            
            # Smart picker selection
            arg_lower = arg_name.lower()
            if any(x in arg_lower for x in ['checkpoint', 'model', 'file']):
                ttk.Button(row, text="...", width=3, command=lambda v=var: self.browse_file(v)).pack(side=tk.LEFT)
            elif any(x in arg_lower for x in ['dir', 'dataset', 'folder', 'path', 'save']):
                ttk.Button(row, text="...", width=3, command=lambda v=var: self.browse_dir(v)).pack(side=tk.LEFT)

    def run_evaluation(self):
        gt_dir = self.gt_dir_var.get()
        if not gt_dir or not os.path.isdir(gt_dir):
            messagebox.showerror("Error", "Valid GT directory required.")
            return

        self.run_btn.config(state=tk.DISABLED)
        self.progress.start()
        
        thread = threading.Thread(target=self.execution_thread, args=(gt_dir,))
        thread.start()

    def execution_thread(self, gt_dir):
        output_base = self.out_dir_var.get()
        
        for model_name in self.selected_model_names:
            self.log(f"--- Starting {model_name} ---\n")
            config = self.model_configs[model_name]
            interface_cls = self.registry[model_name]
            
            # Build arguments dict
            final_args = {k: v.get() for k, v in config["args"].items()}
            
            # Map GT dir
            data_keys = ['input', 'input_dir', 'data', 'dataset']
            target_key = next((k for k in data_keys if k in interface_cls.REQUIRED_ARGS or k in getattr(interface_cls, 'DEFAULT_VARS', {})), "dataset")
            final_args[target_key] = gt_dir
            
            # Output handling
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
                # --- SUBPROCESS PATCHING (Streaming Output) ---
                import subprocess as sp_module
                orig_run = sp_module.run
                orig_check_call = sp_module.check_call

                def streaming_executor(cmd, **kwargs):
                    # Filter out arguments not supported by Popen
                    kwargs.pop('check', None)
                    kwargs.pop('capture_output', None)
                    
                    # Force output to pipe for streaming
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
                    if ret != 0:
                        raise sp_module.CalledProcessError(ret, cmd)
                    return 0

                def custom_run(cmd, *args, **kwargs):
                    ret = streaming_executor(cmd, **kwargs)
                    return sp_module.CompletedProcess(cmd, ret)

                # Apply patches
                sp_module.run = custom_run
                sp_module.check_call = custom_check_call

                # Patch print to log
                original_print = builtins.print
                def custom_print(*args, **kwargs):
                    msg = " ".join(map(str, args))
                    end = kwargs.get('end', '\n')
                    self.log(msg + end)
                builtins.print = custom_print
                
                # Patch input to avoid Tcl panic on background threads
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
                
                # Restore
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
        gt_dir = self.gt_dir_var.get()
        if os.path.isdir(gt_dir):
            imgs = sorted([os.path.basename(f) for f in glob.glob(os.path.join(gt_dir, "*")) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            self.img_selector['values'] = imgs
            if imgs: self.img_selector.current(0)

    def update_comparison(self, event=None):
        selected_img = self.img_selector.get()
        model_name = self.model_selector.get()
        gt_dir = self.gt_dir_var.get()
        out_base = self.out_dir_var.get()
        
        if not selected_img or not model_name: return
        
        gt_path = os.path.join(gt_dir, selected_img)
        model_out = os.path.join(out_base, model_name)
        
        possible = [
            os.path.join(model_out, selected_img),
            os.path.join(model_out, f"rec_{selected_img}"),
            os.path.join(model_out, selected_img.replace(".jpg", ".png").replace(".jpeg", ".png")),
        ]
        
        if model_name == "ELIC":
            elic_dir = os.path.join(ROOT_DIR, "LIC-Models/ELIC/experiments", f"gui_eval_{model_name}", "codestream")
            if os.path.exists(elic_dir):
                epochs = sorted(os.listdir(elic_dir))
                if epochs:
                    possible.append(os.path.join(elic_dir, epochs[-1], selected_img))
                    possible.append(os.path.join(elic_dir, epochs[-1], selected_img.replace(".jpg", ".png").replace(".jpeg", ".png")))

        if model_name == "DCVC-RT":
            base = os.path.splitext(selected_img)[0]
            if os.path.exists(model_out):
                for f in os.listdir(model_out):
                    if f.startswith(f"{base}_q") and f.endswith(".png"):
                        possible.append(os.path.join(model_out, f))
                        break

        found = next((p for p in possible if os.path.exists(p)), None)
        if found:
            self.comp_canvas.set_images(gt_path, found)
        else:
            self.log(f"Reconstructed image not found for {selected_img} in {model_name}")

if __name__ == "__main__":
    root = tk.Tk()
    app = LICApp(root)
    root.mainloop()
