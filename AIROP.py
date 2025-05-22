import sys
import os
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from datetime import datetime

def resource_path(relative_path):
    # Get absolute path to resource, works for dev and for PyInstaller
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

# Load models with resource_path
xception_model = tf.keras.models.load_model(resource_path("xception_rop_model.h5"))
densenet_model = tf.keras.models.load_model(resource_path("densenet_rop_model.h5"))
inception_model = tf.keras.models.load_model(resource_path("inceptionv3_rop_model.h5"))

# Define class labels
model_labels = ["Healthy", "Retinal Detachment", "ROP"]

class FundusClassifierApp:
    def __init__(self, root):
        self.root = root
        try:
            icon_path = resource_path('airop_icon.png')
            icon = tk.PhotoImage(file=icon_path)
            self.root.iconphoto(True, icon)
        except Exception as e:
            print(f"Icon loading error: {e}")
            
        self.root.title("AI-ROP: Fundus Image Classifier")
        self.root.geometry("550x700")
        self.root.minsize(450, 600)
        self.root.config(bg="#e6f7ff")

        self.bg_color = "#e6f7ff"
        self.primary_color = "#0077b6"
        self.secondary_color = "#00b4d8"
        self.accent_color = "#90e0ef"
        self.text_color = "#333333"
        self.highlight_color = "#caf0f8"

        self.main_frame = tk.Frame(root, bg=self.bg_color)
        self.main_frame.pack(fill='both', expand=True, padx=10, pady=10)

        self.title_label = tk.Label(self.main_frame, 
                                    text="AI-ROP: Fundus Image Classifier", 
                                    font=("Arial", 20, "bold"), 
                                    bg=self.bg_color, 
                                    fg=self.primary_color)
        self.title_label.pack(pady=(0, 10), fill='x')

        self.upload_button = tk.Button(self.main_frame, 
                                       text="Upload Image", 
                                       font=("Arial", 14), 
                                       command=self.upload_image, 
                                       bg=self.secondary_color, 
                                       fg="white", 
                                       activebackground=self.primary_color,
                                       activeforeground="white",
                                       width=15, 
                                       relief=tk.RAISED, 
                                       bd=2)
        self.upload_button.pack(pady=10)

        self.image_frame = tk.Frame(self.main_frame, bg=self.highlight_color, bd=2, relief=tk.SUNKEN)
        self.image_frame.pack(pady=10, fill='both', expand=True)
        self.image_label = tk.Label(self.image_frame, bg=self.highlight_color)
        self.image_label.pack(pady=10, padx=10, fill='both', expand=True)

        self.confidence_frame = tk.LabelFrame(self.main_frame, 
                                              text="Confidence Level", 
                                              font=("Arial", 12, "bold"),
                                              bg=self.bg_color, 
                                              fg=self.primary_color,
                                              relief=tk.GROOVE, 
                                              bd=2)
        self.confidence_frame.pack(fill='x', pady=(5, 10), padx=5)

        style = ttk.Style()
        style.theme_use('clam')
        style.configure("blue.Horizontal.TProgressbar",
                        background=self.secondary_color,
                        troughcolor=self.highlight_color,
                        bordercolor=self.primary_color,
                        lightcolor=self.accent_color,
                        darkcolor=self.primary_color)

        self.confidence_bar = ttk.Progressbar(self.confidence_frame, 
                                              orient='horizontal', 
                                              length=200, 
                                              mode='determinate',
                                              style="blue.Horizontal.TProgressbar")
        self.confidence_bar.pack(pady=5, padx=10, fill='x', expand=True)

        self.confidence_label = tk.Label(self.confidence_frame, 
                                         text="0%", 
                                         font=("Arial", 10), 
                                         bg=self.bg_color, 
                                         fg=self.primary_color)
        self.confidence_label.pack(pady=(0, 5))

        self.results_frame = tk.LabelFrame(self.main_frame, 
                                           text="Classification Results", 
                                           font=("Arial", 12, "bold"),
                                           bg=self.bg_color, 
                                           fg=self.primary_color,
                                           relief=tk.GROOVE, 
                                           bd=2)
        self.results_frame.pack(fill='both', pady=5, padx=5, expand=True)

        self.healthy_var = tk.IntVar()
        self.unhealthy_var = tk.IntVar()
        self.rop_var = tk.IntVar()
        self.rd_var = tk.IntVar()

        checkbox_style = {
            "font": ("Arial", 12),
            "bg": self.bg_color,
            "activebackground": self.bg_color,
            "selectcolor": self.highlight_color,
            "bd": 0,
            "highlightthickness": 0
        }

        self.healthy_check = tk.Checkbutton(self.results_frame, text="Healthy Retina", variable=self.healthy_var, fg="#2a9d8f", **checkbox_style)
        self.healthy_check.pack(anchor='w', padx=20, pady=5)

        self.unhealthy_check = tk.Checkbutton(self.results_frame, text="Unhealthy Retina", variable=self.unhealthy_var, fg="#e76f51", **checkbox_style)
        self.unhealthy_check.pack(anchor='w', padx=20, pady=5)

        tk.Frame(self.results_frame, height=1, bg=self.secondary_color).pack(fill='x', padx=10, pady=5)

        self.rop_check = tk.Checkbutton(self.results_frame, text="Retinopathy of Prematurity (ROP)", variable=self.rop_var, fg="#9d4edd", **checkbox_style)
        self.rop_check.pack(anchor='w', padx=20, pady=5)

        self.rd_check = tk.Checkbutton(self.results_frame, text="Retinal Detachment", variable=self.rd_var, fg="#3a86ff", **checkbox_style)
        self.rd_check.pack(anchor='w', padx=20, pady=5)

        current_year = datetime.now().year
        self.copyright_label = tk.Label(self.main_frame, 
                                        text=f"© {current_year} Maryam Saqib, Syeda Aatika Abid Gellani, Shais ur Rehman\nNational University of Computer and Emerging Sciences, Lahore",
                                        font=("Arial", 8), 
                                        bg=self.bg_color, 
                                        fg=self.primary_color)
        self.copyright_label.pack(side='bottom', pady=(10, 0))

        self.disable_all_checkboxes()

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
        if file_path:
            image = Image.open(file_path)
            image.thumbnail((300, 300))
            img = ImageTk.PhotoImage(image)

            self.image_label.configure(image=img)
            self.image_label.image = img

            self.reset_checkboxes()
            self.classify_fundus_image(file_path)

    def reset_checkboxes(self):
        self.healthy_var.set(0)
        self.unhealthy_var.set(0)
        self.rop_var.set(0)
        self.rd_var.set(0)
        self.confidence_bar['value'] = 0
        self.confidence_label.config(text="0%")
        self.enable_all_checkboxes()

    def disable_all_checkboxes(self):
        for checkbox in [self.healthy_check, self.unhealthy_check, self.rop_check, self.rd_check]:
            checkbox.config(state=tk.DISABLED)

    def enable_all_checkboxes(self):
        for checkbox in [self.healthy_check, self.unhealthy_check, self.rop_check, self.rd_check]:
            checkbox.config(state=tk.NORMAL)

    def classify_fundus_image(self, file_path):
        img = Image.open(file_path).resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predictions from all models
        x_pred = xception_model.predict(img_array)[0]
        i_pred = inception_model.predict(img_array)[0]
        d_pred = densenet_model.predict(img_array)[0]

        # Weighted ensemble prediction
        if np.max(x_pred) < 0.9:  # If Xception is unsure (confidence < 90%)
            # Blend Xception (70%) + InceptionV3 (20%) + DenseNet (10%)
            final_pred = 0.7 * x_pred + 0.2 * i_pred + 0.1 * d_pred
        else:
            # Default: Use Xception alone (100% weight)
            final_pred = x_pred
            
        predicted_label = np.argmax(final_pred)
        confidence = np.max(final_pred) * 100

        self.confidence_bar['value'] = confidence
        self.confidence_label.config(text=f"{confidence:.1f}%")

        if predicted_label == 0:
            self.healthy_var.set(1)
            for cb in [self.unhealthy_check, self.rop_check, self.rd_check]:
                cb.config(state=tk.DISABLED)
        elif predicted_label == 1:
            self.rd_var.set(1)
            self.unhealthy_var.set(1)
            for cb in [self.healthy_check, self.rop_check]:
                cb.config(state=tk.DISABLED)
        elif predicted_label == 2:
            self.rop_var.set(1)
            self.unhealthy_var.set(1)
            for cb in [self.healthy_check, self.rd_check]:
                cb.config(state=tk.DISABLED)

if __name__ == "__main__":
    root = tk.Tk()
    app = FundusClassifierApp(root)
    root.mainloop()
