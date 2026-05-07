import os
import cv2
import numpy as np

# Age classes
AGE_BUCKETS = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
               '(25-32)', '(38-43)', '(48-53)', '(60-100)']

# Base directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Model paths
AGE_PROTO = os.path.join(BASE_DIR, "models", "age_deploy.prototxt")
AGE_MODEL = os.path.join(BASE_DIR, "models", "age_net.caffemodel")

# Load model (only once)
age_net = cv2.dnn.readNetFromCaffe(AGE_PROTO, AGE_MODEL)


def predict_age(face_img):
    """
    Predict age from face image
    """

    try:
        blob = cv2.dnn.blobFromImage(
            face_img,
            1.0,
            (227, 227),
            (78.4263377603, 87.7689143744, 114.895847746),
            swapRB=False
        )

        age_net.setInput(blob)
        preds = age_net.forward()
        age = AGE_BUCKETS[preds[0].argmax()]

        return age

    except Exception as e:
        print("Age prediction error:", e)
        return "Unknown"