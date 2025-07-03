import streamlit as st
import cv2
import os
import time
import pandas as pd
from deepface import DeepFace

st.set_page_config(page_title="Real-Time Emotion Detection", layout="centered")

st.title("Real-Time Emotion Detection with Capture Feature")

# Prepare haarcascade path and load classifier
haarcascade_path = os.path.join(os.path.dirname(cv2.__file__), 'data', 'haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier(haarcascade_path)

capture_dir = "captures"
os.makedirs(capture_dir, exist_ok=True)

# Initialize session state variables
if 'running' not in st.session_state:
    st.session_state.running = False
if 'frame_count' not in st.session_state:
    st.session_state.frame_count = 0
if 'emotions_in_frame' not in st.session_state:
    st.session_state.emotions_in_frame = []
if 'capture_log' not in st.session_state:
    st.session_state.capture_log = []

# Buttons for control
run_btn = st.button("Start Camera") if not st.session_state.running else st.button("Stop Camera")
capture_btn = st.button("Capture Frame", disabled=not st.session_state.running)

if run_btn:
    st.session_state.running = not st.session_state.running
    if not st.session_state.running:
        st.session_state.emotions_in_frame = []

frame_window = st.empty()

if st.session_state.running:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Cannot open webcam")
        st.stop()

    try:
        while st.session_state.running:
            ret, frame = cap.read()
            if not ret:
                st.warning("Failed to read frame from webcam")
                break

            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=6, minSize=(30,30))

            st.session_state.frame_count += 1
            if st.session_state.frame_count % 10 == 0:
                emotions = []
                for (x, y, w, h) in faces:
                    face_img = frame[y:y+h, x:x+w]
                    try:
                        result = DeepFace.analyze(face_img, actions=['emotion'], enforce_detection=False)
                        dominant_emotion = result[0]['dominant_emotion']
                    except Exception:
                        dominant_emotion = "Unknown"
                    emotions.append((x, y, w, h, dominant_emotion))
                st.session_state.emotions_in_frame = emotions

            for (x, y, w, h, emotion) in st.session_state.emotions_in_frame:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0,255,0), 2)
                cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_window.image(rgb_frame)

            if capture_btn:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"capture_{timestamp}.jpg"
                filepath = os.path.join(capture_dir, filename)
                cv2.imwrite(filepath, frame)
                
                # Log all detected emotions in this frame
                for idx, (x, y, w, h, emotion) in enumerate(st.session_state.emotions_in_frame):
                    st.session_state.capture_log.append({
                        "Timestamp": timestamp,
                        "Filename": filename,
                        "FaceIndex": idx,
                        "X": x, "Y": y, "W": w, "H": h,
                        "DominantEmotion": emotion
                    })
                st.success(f"Captured frame saved as {filename}")

    finally:
        cap.release()
else:
    frame_window.text("Camera stopped. Click 'Start Camera' to begin.")

# Show capture log table if available
if st.session_state.capture_log:
    st.subheader("Captured Emotions Log")
    df_log = pd.DataFrame(st.session_state.capture_log)
    st.dataframe(df_log)
