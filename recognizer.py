# recognizer.py
import cv2
import numpy as np
import os

def train_model():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    dataset_path = "dataset"
    image_paths = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path)]

    faces = []
    ids = []

    for image_path in image_paths:
        gray_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        label = image_path.split("/")[-1].split("_")[0]  # e.g., Prafful_0.jpg → Prafful
        label_id = hash(label) % 10000  # simple way to create numeric ID

        faces.append(gray_img)
        ids.append(label_id)

    recognizer.train(faces, np.array(ids))
    recognizer.save("trainer.yml")  # Save trained model

    print("✅ Model trained and saved as 'trainer.yml'")
