import cv2
import numpy as np
from PIL import Image
import os

def train_model():
    path = 'dataset'
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def get_images_and_labels(path):
        image_paths = [os.path.join(path, f) for f in os.listdir(path)]
        face_samples = []
        ids = []

        for image_path in image_paths:
            gray_img = Image.open(image_path).convert('L')  # convert to grayscale
            img_np = np.array(gray_img, 'uint8')
            name = os.path.split(image_path)[-1].split("_")[0]  # e.g., Prafful_0.jpg → "Prafful"

            faces = detector.detectMultiScale(img_np)
            for (x, y, w, h) in faces:
                face_samples.append(img_np[y:y+h, x:x+w])
                ids.append(1)  # For now use static ID = 1

        return face_samples, ids

    print("[INFO] Training faces. It will take a few seconds...")
    faces, ids = get_images_and_labels(path)
    recognizer.train(faces, np.array(ids))
    recognizer.save('trainer.yml')
    print(f"[INFO] {len(np.unique(ids))} face(s) trained. Trainer saved as 'trainer.yml'.")

# Call the function
train_model()
