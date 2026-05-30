import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path
import training
import tracking
import plotting
import preprocessing
import pandas as pd


class App(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Computer Vision App")
        self.geometry("800x600")

        container = tk.Frame(self)
        container.pack(fill="both", expand=True)

        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}

        for ScreenClass in (MainMenu, PreProcessingFrame, TrainingFrame, TrackingFrame, PlottingFrame, CrowdAnalyticsFrame):
            frame = ScreenClass(parent=container, controller=self)
            name = ScreenClass.__name__
            self.frames[name] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame("MainMenu")

    def show_frame(self, name: str):
        frame = self.frames[name]
        frame.tkraise()


class MainMenu(tk.Frame):
    def __init__(self, parent, controller: App):
        super().__init__(parent)
        self.controller = controller

        tk.Label(self, text="Main Menu", font=("Arial", 18, "bold", "underline")).pack(pady=40)

        tk.Button(
            self,
            text="Preprocess",
            command=lambda: controller.show_frame("PreProcessingFrame"),
            width=20,
            font=("Arial", 12)
        ).pack(pady=20)

        tk.Button(
            self,
            text="Train",
            command=lambda: controller.show_frame("TrainingFrame"),
            width=20,
            font=("Arial", 12)
        ).pack(pady=20)

        tk.Button(
            self,
            text="Track",
            command=lambda: controller.show_frame("TrackingFrame"),
            width=20,
            font=("Arial", 12)
        ).pack(pady=20)

        tk.Button(
            self,
            text="Plot",
            command=lambda: controller.show_frame("PlottingFrame"),
            width=20,
            font=("Arial", 12)
        ).pack(pady=20)

        tk.Button(
            self,
            text="Crowd Analytics",
            command=lambda: controller.show_frame("CrowdAnalyticsFrame"),
            width=20,
            font=("Arial", 12)
        ).pack(pady=20)

class PreProcessingFrame(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller  # main app (so you can navigate if needed)

        # preprocessing window title
        title_label = tk.Label(self, text="Preprocessing video", font=("Arial", 14, "bold"))
        title_label.pack(pady=10)

        # select video button
        self.vid_label = tk.Label(self, text="Video: (none selected yet)", wraplength=650)
        self.vid_label.pack(pady=5)

        vid_button = tk.Button(self, text="Choose video", command=self.choose_vid)
        vid_button.pack(pady=5)

        # output folder button
        self.folder_label = tk.Label(self, text="Output folder: (none selected yet)", wraplength=650)
        self.folder_label.pack(pady=5)

        folder_button = tk.Button(self, text="Choose output folder", command=self.choose_output_folder)
        folder_button.pack(pady=5)

        # crop button
        crop_button = tk.Button(self, text="Crop", font=("Segoe UI", 9, "bold"), command=self.crop)
        crop_button.pack(pady=40)

        # back to main menu button
        tk.Button(self, text="Back to Main Menu", command=lambda: controller.show_frame("MainMenu")).place(x=10, y=10)

    def choose_vid(self):
        path = filedialog.askopenfilename(
            title="Select video file",
            filetypes=[("mp4 files", "*.mp4"), ("All files", "*.*")]
        )
        if not path:
            return  # user cancelled
        self.video_path = path
        self.vid_label.config(text=f"Video: {path}")

    def choose_output_folder(self):
        path = filedialog.askdirectory(title="Select output folder")
        if not path:
            return  # user cancelled
        self.output_folder_path = path
        self.folder_label.config(text=f"Output folder: {path}")

    def crop(self):
        missing = []

        if not hasattr(self, "video_path") or not self.video_path:
            missing.append("video")

        if not hasattr(self, "output_folder_path") or not self.output_folder_path:
            missing.append("output folder")

        if missing:
            msg = "Please select the following before cropping:\n- " + "\n- ".join(missing)
            messagebox.showerror("Missing inputs", msg)
            return

        preprocessing.ROI(self.video_path, self.output_folder_path)

class PlottingFrame(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller  # main app (so you can navigate if needed)

        title_label = tk.Label(self, text="Choose csv file", font=("Arial", 14, "bold"))
        title_label.pack(pady=10)

        #select csv button
        self.csv_label = tk.Label(self, text="csv: (none selected yet)", wraplength=650)
        self.csv_label.pack(pady=5)

        csv_button = tk.Button(self, text="Choose csv", command=self.choose_csv)
        csv_button.pack(pady=5)

        #select plot button
        plot_button = tk.Button(self, text="Plot", command=self.plot_csv)
        plot_button.pack(pady=5)

        #back to main menu buton
        tk.Button(self, text="Back to Main Menu", command=lambda: controller.show_frame("MainMenu")).place(x= 10, y = 10)

    # -------- Button callbacks --------
    def choose_csv(self):
        path = filedialog.askopenfilename(
            title="Select csv file",
            filetypes=[("csv files", "*.csv"), ("All files", "*.*")])    
        if not path:
            return  # user cancelled
        self.csv_path = Path(path)
        self.csv_label.config(text=f"csv: {path}")


    def plot_csv(self):
        cols = pd.read_csv(self.csv_path, nrows=0).columns 
        has_centers = ("cx_smooth" in cols) and ("cy_smooth" in cols)
        if has_centers:
            df = pd.read_csv(self.csv_path)
            plotting.animate_centers(df, x_col="cx_smooth", y_col="cy_smooth")
        else:
            plotting.animate_centers(plotting.compute_centers(self.csv_path))



class TrackingFrame(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller  # main app (so you can navigate if needed)

        # self.selected_pt_path = None
        # self.selected_input_dir = None
        # self.selected_output_tracked_dir = None

        #tracking window title
        title_label = tk.Label(self, text="Tracking", font=("Arial", 14, "bold"))
        title_label.pack(pady=10)

        #select weight button
        self.pt_label = tk.Label(self, text="Weight: (none selected yet)", wraplength=650)
        self.pt_label.pack(pady=5)

        pt_button = tk.Button(self, text="Choose .pt file", command=self.choose_pt)
        pt_button.pack(pady=5)

        #input mp4 button
        self.input_folder_label = tk.Label(self, text="Input: (none selected yet)", wraplength=650)
        self.input_folder_label.pack(pady=5)

        input_folder_button = tk.Button(self, text="Choose .mp4 file", command=self.choose_mp4)
        input_folder_button.pack(pady=5)    

        #output folder button
        self.output_folder_label = tk.Label(self, text="Output: (none selected yet)", wraplength=650)
        self.output_folder_label.pack(pady=5)

        output_folder_button = tk.Button(self, text="Choose folder", command=self.choose_output_folder)
        output_folder_button.pack(pady=5)    
        
        #run tracking button
        start_tracking_button = tk.Button(self, text="Start tracking", font = ("Segoe UI",9,"bold"), command=self.track_mp4)
        start_tracking_button.pack(pady=40)
        

        #back to main menu buton
        tk.Button(self, text="Back to Main Menu", command=lambda: controller.show_frame("MainMenu")).place(x= 10, y = 10)

    # -------- Button callbacks --------
    def choose_pt(self):
        path = filedialog.askopenfilename(
            title="Select model .pt file",
            filetypes=[("pt files", "*.pt"), ("All files", "*.*")])    
        if not path:
            return  # user cancelled
        self.model_path = path
        self.pt_label.config(text=f"Weight: {path}")

    def choose_mp4(self):
        path = filedialog.askopenfilename(
            title="Select video to track",
            filetypes=[("pt files", "*.mp4"), ("All files", "*.*")])
        if not path:
            return  # user cancelled
        self.input_video_path = path
        self.input_folder_label.config(text=f"Input: {path}")

    def choose_output_folder(self):
        path = filedialog.askdirectory(
            title="Select Output folder")
        if not path:
            return  # user cancelled
        self.output_folder_path = path
        self.output_folder_label.config(text=f"Output: {path}")

    def track_mp4(self):
        missing = []
        # Check for required attributes and whether they are non-empty
        if not hasattr(self, "input_video_path") or not self.input_video_path:
            missing.append("input video")

        if not hasattr(self, "output_folder_path") or not self.output_folder_path:
            missing.append("output folder")

        if not hasattr(self, "model_path") or not self.model_path:
            missing.append("model file (.pt)")

        if missing:
            # Build a nice error message
            msg = "Please select the following before tracking:\n- " + "\n- ".join(missing)
            messagebox.showerror("Missing inputs", msg)
            return  # stop here, don't call tracking
        
        input_path = self.input_video_path
        output_path = self.output_folder_path
        model_path = self.model_path
        tracking.track_mp4(model_path, input_path, output_path)

class TrainingFrame(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller  # main app (so you can navigate if needed)

        self.selected_yaml_path = None
        self.selected_output_dir = None

        # --- UI widgets (same as your old root-based ones, but on self) ---
        title_label = tk.Label(self, text="Simple YOLO Training UI", font=("Arial", 14, "bold"))
        title_label.pack(pady=10)

        self.yaml_label = tk.Label(self, text="YAML: (none selected yet)", wraplength=650)
        self.yaml_label.pack(pady=5)

        yaml_button = tk.Button(self, text="Choose .yaml file", command=self.choose_yaml)
        yaml_button.pack(pady=5)

        self.output_label = tk.Label(self, text="Output folder: (none selected yet)", wraplength=650)
        self.output_label.pack(pady=5)

        output_button = tk.Button(self, text="Choose output folder", command=self.choose_output_folder)
        output_button.pack(pady=5)

        # Epochs
        tk.Label(self, text="Epochs:").pack()
        self.epochs_var = tk.StringVar(value="50")  # default
        self.epochs_combo = ttk.Combobox(
            self,
            textvariable=self.epochs_var,
            values=["10", "20", "50", "100", "200"],
            state="readonly",  # user must pick from list
        )
        self.epochs_combo.pack(pady=3)

        self.train_button = tk.Button(self, text="Train model", font = ("Segoe UI",9,"bold"), command=self.start_training)
        self.train_button.pack(pady=15)

        #back to main menu buton
        tk.Button(self, text="Back to Main Menu", command=lambda: controller.show_frame("MainMenu")).place(x= 10, y = 10)

        # -------- Button callbacks --------
    def choose_yaml(self):
        path = filedialog.askopenfilename(
            title="Select data .yaml file",
            filetypes=[("YAML files", "*.yaml *.yml"), ("All files", "*.*")]
        )

        if not path:
            return  # user cancelled

        self.selected_yaml_path = path
        self.yaml_label.config(text=f"YAML: {path}")


    def choose_output_folder(self):
        global selected_output_dir
        path = filedialog.askdirectory(
            title="Select output folder for training runs"
        )

        if not path:
            return  # user cancelled

        self.selected_output_dir = path
        self.output_label.config(text=f"Output folder: {path}")


    def start_training(self):
        # basic validation
        if not self.selected_yaml_path:
            messagebox.showwarning("Missing YAML", "Please choose a .yaml file first.")
            return

        if not self.selected_output_dir:
            messagebox.showwarning("Missing output folder", "Please choose an output folder.")
            return

        # Optional: ask for confirmation
        confirm = messagebox.askyesno(
            "Start training",
            f"Train with:\n\nYAML:\n{self.selected_yaml_path}\n\nOutput folder:\n{self.selected_output_dir}\n\nContinue?"
        )
        if not confirm:
            return
        epochs_text = self.epochs_var.get().strip()

        try:
            self.train_button.config(state="disabled")
            self.controller.update_idletasks()
            epochs = int(epochs_text)

            training.training_model(data_yaml=self.selected_yaml_path,
                                    project_dir=self.selected_output_dir,
                                    epochs = epochs,)

            messagebox.showinfo(
                "Training complete",
                f"Training finished.\n\nbest.pt saved at:\n{self.selected_output_dir}/train/weights/best.pt"
            )

        except Exception as e:
            messagebox.showerror("Error during training", f"{e}")
        finally:
            self.train_button.config(state="normal")

class CrowdAnalyticsFrame(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller  # main app (so you can navigate if needed)

        title_label = tk.Label(self, text="Choose csv file", font=("Arial", 14, "bold"))
        title_label.pack(pady=10)

        #select csv button
        self.csv_label = tk.Label(self, text="csv: (none selected yet)", wraplength=650)
        self.csv_label.pack(pady=5)

        csv_button = tk.Button(self, text="Choose csv", command=self.choose_csv)
        csv_button.pack(pady=5)

        # -------- Button callbacks --------
    def choose_csv(self):
        path = filedialog.askopenfilename(
            title="Select model csv file",
            filetypes=[("csv files", "*.csv"), ("All files", "*.*")])    
        if not path:
            return  # user cancelled
        self.csv_path = Path(path)
        self.csv_label.config(text=f"csv: {path}")

    # def plot_speed_histogram(self):
    #     plotting.plot_speed_histogram(self.csv_path)

if __name__ == "__main__":
    app = App()
    app.mainloop()

