from playsound import playsound
from tkinter import *
import cv2
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import os

root = Tk()
root.geometry("195x180")
root.config(bg="#001d3d")
root.resizable(False, False)
playsound('startup.mp3')

photo = PhotoImage(file="soda.png")
root.iconphoto(False, photo)

def takePicture():
    cam = cv2.VideoCapture(0)

    result, image = cam.read()

    if result:

        cv2.imshow("SolveAbles Image", image)
        cv2.waitKey(1)

        cv2.imwrite("latestimage.png", image)

        showLabel.config(text="Loading...")

        model = load_model('keras_model.h5')

        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

        image = Image.open('latestimage.png').convert('RGB')

        size = (224, 224)
        image = ImageOps.fit(image, size, Image.ANTIALIAS)

        image_array = np.asarray(image)

        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

        data[0] = normalized_image_array

        class_names = ["Sprite", "Red-Bull", "Coca-Cola", "Fanta", "Canada Dry"]

        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index]
        text = class_name
        showLabel.config(text=text)
        os.remove("latestimage.png")

    else:
        print("No image detected. Please try again")


Picture = Button(root, text='📷', font=(
    'Marvel Bold', 30, 'bold'), relief=GROOVE, height=2, width=8, bg="#ffd60a", activebackground="#ffc300", fg="#023e8a", activeforeground="#023e8a", command=takePicture)
Picture.place(x=25, y=25)

showLabel = Label(root, text="...", font=(
    'Helvetica', 10, 'bold'), bg="#001d3d", fg="#ffd60a")
showLabel.place(x=25, y=0)

root.title("SolveAbles")
root.mainloop()