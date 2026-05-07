import cv2
import numpy as np
from tensorflow.keras.models import load_model

drowsy_model = load_model("models/drowsiness_model.h5")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    sleeping_count = 0
    total_people = len(faces)

    results = []

    for (x,y,w,h) in faces:
        face = frame[y:y+h, x:x+w]
        resized = cv2.resize(face, (24,24))
        resized = resized / 255.0
        resized = resized.reshape(1,24,24,3)

        pred = drowsy_model.predict(resized)

        label = "Awake"
        color = (0,255,0)

        if pred < 0.5:
            label = "Sleeping"
            color = (0,0,255)
            sleeping_count += 1

        results.append((x,y,w,h,label,color))

    return results, total_people, sleeping_count