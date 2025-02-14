import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
import streamlit as st
from datetime import datetime

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# Function to detect faces and mark attendance
def detect_faces(image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_image)

    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = image.shape
            x, y, w, h = (int(bboxC.xmin * iw), int(bboxC.ymin * ih),
                          int(bboxC.width * iw), int(bboxC.height * ih))

            # Draw bounding box around detected face
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Simulated face recognition (assign "Unknown")
            name = "Unknown"

            # Display name
            cv2.putText(image, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    return image

# Streamlit App
st.title("Face Recognition Attendance System (MediaPipe)")

# Upload Image for Face Detection
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    # Process image
    processed_image = detect_faces(image)

    # Show result
    st.image(processed_image, channels="BGR")
