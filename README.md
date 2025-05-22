# AI-ROP
Automated Diagnosis of Retinopathy of Prematurity in Infants using Multi-Level Classification

AI-ROP is an AI-powered diagnostic tool for the early detection and classification of Retinopathy of Prematurity (ROP) in premature infants. It uses deep learning and image processing techniques to analyze retinal fundus images and classify them into three clinically relevant categories:
- **Healthy**
- **ROP (Retinopathy of Prematurity)**
- **RD (Retinal Detachment)**

## Features

- Real-time retinal image classification with confidence scoring
- Weighted ensemble of Xception, DenseNet, and InceptionV3
- Vessel segmentation module for enhanced feature focus
- User-friendly graphical interface using Tkinter
- Robust preprocessing using OpenCV, Pillow, and CLAHE
- Data augmentation and management using RoboFlow

## Technologies Used

- Python
- TensorFlow, Keras
- OpenCV, Pillow
- NumPy
- RoboFlow
- Tkinter (GUI)
- Google Colab, Kaggle, Jupyter Notebook (Model Training)

## System Architecture

- **User Interface (Tkinter)**: Upload image, visualize diagnosis with checkboxes and progress bar
- **Image Processing Module**: Resize, normalize, denoise, contrast enhancement, vessel segmentation
- **Model Integration**: Weighted soft voting among Xception, DenseNet, InceptionV3 for robust predictions

## Application Workflow

1. **Upload Fundus Image** via GUI
2. **Preprocessing**: Resize, normalize, segment vessels
3. **Initial Classification**: Healthy vs Unhealthy
4. **Secondary Classification**: ROP vs Retinal Detachment if Unhealthy
5. **Display Results**: Visual feedback with diagnosis and confidence score

## Use Cases

- **NICUs**: Early detection for prompt treatment
- **Remote Clinics/Telemedicine**: AI-supported diagnosis without specialists
- **Research and National Screening Programs**

## Contributors

- Maryam Saqib
- Syeda Aatika Abid Gellani
- Shais ur Rehman
- Supervisor: Dr. Aamir Wali

## Acknowledgements

Clinical support and dataset provided by:
- Dr. Tayyaba Gul Malik, Head of Ophthalmology, Lahore General Hospital

## License

This project is licensed under the **Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License**.

[![License: CC BY-NC-ND 4.0](https://licensebuttons.net/l/by-nc-nd/4.0/88x31.png)](https://creativecommons.org/licenses/by-nc-nd/4.0/)
