import streamlit as st
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

# Title
st.title("🔥 Face Recognition Attendance System")

# Upload Image for Registration
st.subheader("📸 Register a New Student")
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    # Save the Image
    student_name = st.text_input("Enter Student Name:")
    if st.button("Save Image"):
        if student_name:
            file_path = f"images/{student_name}.jpg"
            cv2.imwrite(file_path, image)
            st.success(f"✅ {student_name} has been registered!")
        else:
            st.warning("⚠ Please enter a name!")

# Face Recognition System
st.subheader("🎥 Start Face Recognition")

if st.button("Start Webcam"):
    cap = cv2.VideoCapture(0)

    # Load all known images
    path = "images"
    images = []
    classNames = []
    for cl in os.listdir(path):
        curImg = cv2.imread(f"{path}/{cl}")
        images.append(curImg)
        classNames.append(os.path.splitext(cl)[0])

    # Encode Faces
    def findEncodings(images):
        encodeList = []
        for img in images:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encode = face_recognition.face_encodings(img)[0]
            encodeList.append(encode)
        return encodeList

    encodeListKnown = findEncodings(images)

    while True:
        success, img = cap.read()
        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                name = classNames[matchIndex].upper()
                st.write(f"✅ {name} Recognized")
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
