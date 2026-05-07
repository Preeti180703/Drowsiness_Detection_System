import streamlit as st
import cv2
import numpy as np
from utils.detect import detect
from utils.age_predict import predict_age
from PIL import Image

st.title("🚗 Drowsiness Detection System with Age Prediction")

option = st.radio("Select Input Type", ["Image", "Video"])

# IMAGE MODE
if option == "Image":
    file = st.file_uploader("Upload Image", type=["jpg","png"])

    if file:
        image = Image.open(file)
        frame = np.array(image)

        results, total, sleeping = detect(frame)

        for (x,y,w,h,label,color) in results:
            face = frame[y:y+h, x:x+w]
            age = predict_age(face)

            cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
            cv2.putText(frame, f"{label}, Age:{age}", (x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        st.image(frame, channels="BGR")

        # POPUP MESSAGE
        if sleeping > 0:
            st.error(f"⚠ ALERT: {sleeping} person(s) sleeping! Ages detected.")
        else:
            st.success("✅ All Awake")

        st.write(f"Total People: {total}")

# VIDEO MODE
elif option == "Video":
    file = st.file_uploader("Upload Video", type=["mp4"])

    if file:
        tfile = open("temp.mp4", "wb")
        tfile.write(file.read())

        cap = cv2.VideoCapture("temp.mp4")

        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results, total, sleeping = detect(frame)

            for (x,y,w,h,label,color) in results:
                face = frame[y:y+h, x:x+w]
                age = predict_age(face)

                cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
                cv2.putText(frame, f"{label}, Age:{age}", (x,y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            stframe.image(frame, channels="BGR")

            if sleeping > 0:
                st.warning(f"⚠ {sleeping} sleeping detected")

        cap.release()