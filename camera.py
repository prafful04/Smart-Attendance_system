import cv2
import os

def start_camera():
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)

    count = 0
    user_id = input("Enter your ID or Name: ")

    dataset_path = 'dataset'
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            # Draw rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Save face image
            face_img = gray[y:y + h, x:x + w]
            file_name = f"{dataset_path}/{user_id}_{count}.jpg"
            cv2.imwrite(file_name, face_img)
            count += 1

        cv2.imshow("Saving Faces - Press Q to Quit", frame)

        if cv2.waitKey(1) & 0xFF == ord('q') or count >= 20:
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Saved {count} face images for user: {user_id}")
