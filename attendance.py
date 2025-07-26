# attendance.py
import cv2
import numpy as np
import os
import csv
from datetime import datetime

# Check if attendance.csv exists, if not, create it with headers
if not os.path.exists('attendance.csv'):
    with open('attendance.csv', mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Name', 'Date', 'Time'])

# Load recognizer and cascade
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Map face ID to name (IMPORTANT: Should match how IDs were generated)
def get_name_from_id(id_value):
    name_map = {
        hash("Prafful") % 10000: "Prafful",  # Add more if needed
    }
    return name_map.get(id_value, "Unknown")

def mark_attendance(name):
    filename = "attendance.csv"
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")

    # Check if file exists
    if not os.path.isfile(filename):
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Name", "Date", "Time"])

    # Check if already marked today
    already_marked = False
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if row and row[0] == name and row[1] == date:
                already_marked = True
                break

    if not already_marked:
        with open(filename, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([name, date, time])
            print(f"✅ Attendance marked for {name} at {time}")

def recognize_faces():
    cam = cv2.VideoCapture(0)

    print("[INFO] Starting face recognition...")
    while True:
        ret, frame = cam.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_img = gray[y:y + h, x:x + w]
            id_, confidence = recognizer.predict(face_img)

            name = get_name_from_id(id_)
            if confidence < 80:
                mark_attendance(name)
                color = (0, 255, 0)
                label = f"{name} ({round(100 - confidence)}%)"
            else:
                color = (0, 0, 255)
                label = "Unknown"

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("Face Recognition - Attendance", frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

def recognize_faces_from_image(filepath):
    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        return

    gray = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_img = gray[y:y + h, x:x + w]
        id_, confidence = recognizer.predict(face_img)

        name = get_name_from_id(id_)
        if confidence < 80:
            mark_attendance(name)
            print(f"✅ Recognized {name} with confidence {round(100 - confidence)}%")
        else:
            print("❌ Unknown face detected")
