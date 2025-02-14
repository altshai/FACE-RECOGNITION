import streamlit as st
import cv2
import numpy as np
import face_recognition
import tempfile
import os

st.title("Face Recognition Attendance System")

# Upload an image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Save uploaded image
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    temp_file.write(uploaded_file.read())

    # Read the uploaded image
    img = cv2.imread(temp_file.name)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Detect faces
    face_locations = face_recognition.face_locations(rgb_img)

    if face_locations:
        st.success("Face detected!")
        for (top, right, bottom, left) in face_locations:
            cv2.rectangle(rgb_img, (left, top), (right, bottom), (0, 255, 0), 2)

        st.image(rgb_img, caption="Detected Face", use_column_width=True)
    else:
        st.warning("No face detected!")

    os.remove(temp_file.name)
