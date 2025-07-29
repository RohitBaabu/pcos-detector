 🧠 PCOS Detector – CNN + Tkinter GUI

An offline desktop application that detects Polycystic Ovary Syndrome (PCOS) using a trained Convolutional Neural Network (CNN). Built with Python, TensorFlow, and Tkinter, this tool allows users to upload an image and instantly check for infection status with a confidence score.

---

 🚀 Features

- ✅ Image-based PCOS detection (infected / not infected)
- ✅ Trained CNN using medical image dataset
- ✅ GUI built with Tkinter for easy use
- ✅ Works 100% offline – no internet needed
- ✅ Shows prediction label and confidence
- ✅ Can be expanded to other medical conditions

---

 📁 Project Structure

pcos-detector/
├── app.py # GUI frontend
├── train.py # CNN training script
├── models/
│ └── best_pcos_model.h5 # Trained model
├── data/  (dataset to test and train model)
│ └──test
│    └──infected
│    └──not_infected
│ └──train
│    └──infected
│    └──not_infected
├── requirements.txt # Required Python packages
└── README.md

yaml
Copy
Edit

---

🛠️ Installation

 1. Install required packages
pip install -r requirements.txt

 2. Train the model (if needed)
python train.py

 3. Run the app
python app.py
