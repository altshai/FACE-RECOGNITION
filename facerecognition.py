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
def detect_faces(frame, students, student_ids):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)

    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x, y, w, h = (int(bboxC.xmin * iw), int(bboxC.ymin * ih),
                          int(bboxC.width * iw), int(bboxC.height * ih))

            # Draw bounding box around detected face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Simulating face recognition (For Demo, assigns "Unknown")
            name = "Unknown"
            if len(student_ids) > 0:
                name = student_ids[0]  # Assign first student ID as recognized face
            
            # Mark Attendance
            mark_attendance(name)
            
            # Display name
            cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    return frame

# Function to mark attendance
def mark_attendance(name):
    df = pd.read_csv('Attendance.csv') if 'Attendance.csv' in os.listdir() else pd.DataFrame(columns=['Name', 'Time'])

    if name not in df['Name'].values:
        now = datetime.now().strftime('%H:%M:%S')
        df = df.append({'Name': name, 'Time': now}, ignore_index=True)
        df.to_csv('Attendance.csv', index=False)

# Streamlit App
st.title("Face Recognition Attendance System (MediaPipe)")

# Upload Image or Use Webcam
option = st.selectbox("Choose Input Method", ["Webcam", "Upload Image"])

if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
        image = detect_faces(image, [], [])
        st.image(image, channels="BGR")

elif option == "Webcam":
    st.write("Click below to start webcam.")
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if ret:
        frame = detect_faces(frame, [], [])
        st.image(frame, channels="BGR")
    cap.release()
