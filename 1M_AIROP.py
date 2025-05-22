import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model("xception_rop_model.h5")

# Define class labels (Only what model predicts)
model_labels = ["Healthy", "Retinal Detachment", "ROP"]

class FundusClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI-ROP: Fundus Image Classifier")
        self.root.geometry("500x600")
        self.root.minsize(400, 500)  # Minimum size
        self.root.config(bg="#f0f4c3")  # Light yellow background

        # Title label
        self.title_label = tk.Label(root, text="AI-ROP: Fundus Image Classifier", font=("Arial", 20, "bold"), bg="#f0f4c3", fg="#388e3c")
        self.title_label.pack(pady=10, fill='x', expand=True)

        # Upload button centered and with small width
        self.upload_button = tk.Button(root, text="Upload Image", font=("Arial", 14), command=self.upload_image, bg="#4caf50", fg="white", width=15)
        self.upload_button.pack(pady=10)  # Centered button with reduced width

        # Image display area
        self.image_label = tk.Label(root, bg="#f0f4c3")
        self.image_label.pack(pady=10, fill='both', expand=True)

        # Classification checkboxes frame with two centered columns
        self.checkboxes_frame = tk.Frame(root, bg="#f0f4c3")
        self.checkboxes_frame.pack(pady=10)

        # Variables for checkboxes
        self.healthy_var = tk.IntVar()
        self.unhealthy_var = tk.IntVar()
        self.rop_var = tk.IntVar()
        self.rd_var = tk.IntVar()

        # First column (Healthy or Unhealthy) - centered
        self.healthy_checkbox = tk.Checkbutton(self.checkboxes_frame, text="Healthy", variable=self.healthy_var, font=("Arial", 14), bg="#f0f4c3", fg="#388e3c", state=tk.DISABLED)
        self.unhealthy_checkbox = tk.Checkbutton(self.checkboxes_frame, text="Unhealthy", variable=self.unhealthy_var, font=("Arial", 14), bg="#f0f4c3", fg="#388e3c", state=tk.DISABLED)

        # Second column (ROP Stages) - centered
        self.rop_checkbox = tk.Checkbutton(self.checkboxes_frame, text="ROP", variable=self.rop_var, font=("Arial", 14), bg="#f0f4c3", fg="#388e3c", state=tk.DISABLED)
        self.rd_checkbox = tk.Checkbutton(self.checkboxes_frame, text="Retinal Detachment", variable=self.rd_var, font=("Arial", 14), bg="#f0f4c3", fg="#388e3c", state=tk.DISABLED)

        # Grid layout: first column for Healthy/Unhealthy, second for types, both centered
        self.healthy_checkbox.grid(row=0, column=0, padx=10, pady=5, sticky="w")
        self.unhealthy_checkbox.grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.rop_checkbox.grid(row=0, column=1, padx=10, pady=5, sticky="w")
        self.rd_checkbox.grid(row=1, column=1, padx=10, pady=5, sticky="w")

        # Center the checkboxes by using "columnspan" to distribute space
        self.checkboxes_frame.grid_columnconfigure(0, weight=1)
        self.checkboxes_frame.grid_columnconfigure(1, weight=1)

    def upload_image(self):
        # File dialog to choose an image file
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
        
        if file_path:
            # Load the image and display it
            image = Image.open(file_path)
            image.thumbnail((300, 300))  # Smaller thumbnail for minimized windows
            img = ImageTk.PhotoImage(image)

            self.image_label.configure(image=img)
            self.image_label.image = img

            # Clear all checkboxes before new classification
            self.reset_checkboxes()

            # Classify the image using the model
            self.classify_fundus_image(file_path)

    def reset_checkboxes(self):
        # Reset all checkbox states
        self.healthy_var.set(0)
        self.unhealthy_var.set(0)
        self.rop_var.set(0)
        self.rd_var.set(0)

    def classify_fundus_image(self, file_path):
        # Load and preprocess image
        img = Image.open(file_path).resize((224, 224))  # Resize for model
        img_array = np.array(img) / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)  # Expand dims for model
        
        # Predict
        prediction = model.predict(img_array)[0]  # Get predictions
        predicted_label = np.argmax(prediction)  # Get highest probability class
        
        # Assign correct checkboxes based on prediction
        if predicted_label == 0:  # Healthy
            self.healthy_var.set(1)  # Healthy

        elif predicted_label == 1:  # Retinal Detachment
            self.rd_var.set(1)  # Retinal Detachment
            self.unhealthy_var.set(1)  # Unhealthy (UI-only)

        elif predicted_label == 2:  # ROP
            self.rop_var.set(1)  # ROP
            self.unhealthy_var.set(1)  # Unhealthy (UI-only)

# Create the main application window
if __name__ == "__main__":
    root = tk.Tk()
    app = FundusClassifierApp(root)
    root.mainloop()