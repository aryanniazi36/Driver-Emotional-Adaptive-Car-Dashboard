import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Circle
import time
import threading
import os

# ------------------- Load Emotion Model -------------------
try:
    model = tf.keras.models.load_model("emotion_model.h5")
except:
    st.error("❌ emotion_model.h5 not found! Make sure it's in the same folder as app_upgraded.py")
    st.stop()

class_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# ------------------- Streamlit Setup -------------------
st.set_page_config(page_title="Driver Adaptive Car Dashboard", layout="wide")
st.title("🚗 Upgraded Driver Emotional & Adaptive Car Dashboard")
st.sidebar.header("⚙️ Settings")

enable_camera = st.sidebar.checkbox("Enable Camera", True)
stress_threshold = st.sidebar.slider("Stress Threshold", 0.1, 1.0, 0.6, 0.1)
drowsy_frames_threshold = st.sidebar.slider("Frames for Sleep Detection", 5, 50, 15)
audio_alert_file = "alert.wav"

resume_drive = st.sidebar.button("▶️ Resume Driving")

# ------------------- Mediapipe Face Mesh -------------------
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# ------------------- Placeholders -------------------
placeholder_cam = st.empty()
placeholder_chart = st.empty()
placeholder_dashboard = st.empty()
placeholder_score = st.empty()

stress_history = []
time_history = []

# ------------------- Audio Alert -------------------
def play_alert_sound():
    if os.path.exists(audio_alert_file):
        threading.Thread(target=lambda: os.system(f'start {audio_alert_file}'), daemon=True).start()

# ------------------- Eye Aspect Ratio -------------------
def eye_aspect_ratio(landmarks, eye_indices, frame_w, frame_h):
    points = [(int(landmarks[idx].x * frame_w), int(landmarks[idx].y * frame_h)) for idx in eye_indices]
    A = np.linalg.norm(np.array(points[1]) - np.array(points[5]))
    B = np.linalg.norm(np.array(points[2]) - np.array(points[4]))
    C = np.linalg.norm(np.array(points[0]) - np.array(points[3]))
    return (A + B) / (2.0 * C)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
EAR_THRESHOLD = 0.25

# ------------------- Camera Setup -------------------
cap = cv2.VideoCapture(0)
start_time = time.time()
drowsy_counter = 0
car_speed = 60
brake_engaged = False
driver_score = 100  # Initial score

# ------------------- Speedometer Function -------------------
def draw_speedometer(speed, max_speed=100):
    fig, ax = plt.subplots(figsize=(4,4))
    ax.axis('equal')
    wedge = Wedge(center=(0,0), r=1, theta1=180, theta2=0, facecolor='lightgrey', edgecolor='black', lw=2)
    ax.add_patch(wedge)
    angle = 180 - (speed/max_speed)*180
    ax.plot([0, np.cos(np.radians(angle))], [0, np.sin(np.radians(angle))], lw=4, color='red')
    ax.add_patch(Circle((0,0), 0.05, color='black'))
    ax.set_xlim(-1.2,1.2)
    ax.set_ylim(0,1.2)
    ax.axis('off')
    ax.set_title(f"Speed: {speed:.1f} km/h")
    return fig

# ------------------- Main Loop -------------------
while enable_camera:
    ret, frame = cap.read()
    if not ret:
        st.error("❌ Camera not detected!")
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    emotion = "Neutral"
    stress_level = 0.3
    drowsy_alert = False

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # ------------------- Emotion Detection -------------------
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w, _ = frame.shape
            resized = cv2.resize(gray_frame, (48,48))
            norm_img = np.expand_dims(resized / 255.0, axis=(0,-1))
            prediction = model.predict(norm_img, verbose=0)
            emotion = class_labels[np.argmax(prediction)]
            stress_level = float(np.max(prediction))

            # ------------------- Drowsiness Detection -------------------
            left_ear = eye_aspect_ratio(face_landmarks.landmark, LEFT_EYE, w, h)
            right_ear = eye_aspect_ratio(face_landmarks.landmark, RIGHT_EYE, w, h)
            avg_ear = (left_ear + right_ear)/2.0
            if avg_ear < EAR_THRESHOLD:
                drowsy_counter += 1
            else:
                drowsy_counter = 0
            if drowsy_counter >= drowsy_frames_threshold:
                drowsy_alert = True

    # ------------------- Stress History -------------------
    stress_history.append(stress_level)
    time_history.append(time.time() - start_time)
    stress_avg = np.mean(stress_history[-10:])

    # ------------------- Auto-Brake Logic -------------------
    brake_alert = False
    if (stress_avg > stress_threshold) or drowsy_alert:
        if not brake_engaged:
            car_speed = 0
            brake_engaged = True
            brake_alert = True
            car_status = "🚨 Automatic Brake Engaged due to High Stress/Drowsiness!"
            play_alert_sound()
    elif resume_drive:
        brake_engaged = False
        car_status = "▶️ Driving Resumed"
        brake_alert = False
    else:
        if not brake_engaged:
            if stress_level > stress_threshold:
                car_status = "⚠️ High Stress! Adaptive measures simulated!"
                car_speed -= 1
            else:
                car_status = "✅ Normal Driving Mode"
                car_speed = min(car_speed + 0.5, 60)
        if drowsy_alert:
            car_status += " | 😴 Drowsy detected! Take a break!"
            brake_alert = True
            car_speed = max(car_speed-2, 0)

    # ------------------- Driver Gamification Score -------------------
    if car_speed == 60 and stress_level < stress_threshold:
        driver_score = min(driver_score + 0.1, 100)
    elif brake_engaged:
        driver_score = max(driver_score - 1, 0)

    # ------------------- Display Camera -------------------
    placeholder_cam.image(frame, channels="BGR")

    # ------------------- Stress Graph -------------------
    fig, ax = plt.subplots()
    ax.plot(time_history, stress_history, label="Stress Level")
    ax.axhline(y=stress_threshold, color='r', linestyle='--', label="Stress Threshold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Stress Level")
    ax.legend()
    placeholder_chart.pyplot(fig)

    # ------------------- Dashboard -------------------
    dashboard_html = f"""
    <div style='padding:10px; border:2px solid black; border-radius:10px; width:450px; background:#f0f0f0'>
        <h3>🚗 Upgraded Adaptive Car Dashboard</h3>
        <p><b>Emotion:</b> {emotion}</p>
        <p><b>Stress Level:</b> {stress_level:.2f} {'🔴' if stress_level>stress_threshold else '🟢'}</p>
        <p><b>Speed:</b> {car_speed:.1f} km/h</p>
        <p><b>Status:</b> {car_status}</p>
        <p style='color:red;'>{'🚨 Brake Alert!' if brake_alert else ''}</p>
        <p style='color:orange;'>{'⚠️ Emergency Brake Active!' if brake_engaged else ''}</p>
    </div>
    """
    placeholder_dashboard.markdown(dashboard_html, unsafe_allow_html=True)

    # ------------------- Speedometer -------------------
    fig_speed = draw_speedometer(car_speed)
    st.pyplot(fig_speed)

    # ------------------- Driver Gamification Score -------------------
    placeholder_score.markdown(f"<h3>🏆 Driver Score: {driver_score:.1f}/100</h3>", unsafe_allow_html=True)
