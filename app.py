import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf

IMG_SIZE = 128
MODEL_PATH = 'models/best_pcos_model.h5'

model = tf.keras.models.load_model(MODEL_PATH)

# GUI
window = tk.Tk()
window.title("PCOS Detection AI")
window.geometry("400x500")

label_result = tk.Label(window, text="", font=("Arial", 16))
label_result.pack(pady=20)

canvas = tk.Canvas(window, width=IMG_SIZE, height=IMG_SIZE)
canvas.pack()

def predict_image(image_path):
    img = tf.keras.utils.load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    prediction = model.predict(img_array)[0][0]

    # FIXED: Flip logic based on class index {'infected': 0, 'not_infected': 1}
    if prediction < 0.5:
        result = "Infected"
        confidence = 1 - prediction
    else:
        result = "Not Infected"
        confidence = prediction

    label_result.config(text=f"{result} ({confidence*100:.2f}% confidence)")

def load_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = Image.open(file_path)
        img = img.resize((IMG_SIZE, IMG_SIZE))
        img_tk = ImageTk.PhotoImage(img)
        canvas.image = img_tk
        canvas.create_image(0, 0, anchor="nw", image=img_tk)
        predict_image(file_path)

btn_upload = tk.Button(window, text="Upload Image", command=load_image)
btn_upload.pack(pady=10)

window.mainloop()
