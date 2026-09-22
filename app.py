import streamlit as st
import cv2
import os
import pandas as pd
import threading
from datetime import datetime
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av


# =========================
# PATHS
# =========================

base_dir = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(
    base_dir, "trainer", "trainer.yml"
)

cascade_path = os.path.join(
    base_dir, "models", "haarcascade_frontalface_default.xml"
)

attendance_file = os.path.join(
    base_dir, "attendance", "attendance.csv"
)


# =========================
# STUDENT NAMES
# =========================

student_names = {
    101: "Ibrahim",
    102: "Tanveer",
    103: "Baba",
    104: "Revanth"
}


# =========================
# LOAD FACE RECOGNITION
# =========================

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(model_path)

face_detector = cv2.CascadeClassifier(cascade_path)

file_lock = threading.Lock()


# =========================
# MARK ATTENDANCE
# =========================

def mark_attendance(student_id, name):

    today = datetime.now().strftime("%Y-%m-%d")
    current_time = datetime.now().strftime("%H:%M:%S")

    os.makedirs(
        os.path.dirname(attendance_file),
        exist_ok=True
    )

    with file_lock:

        # Create file if it doesn't exist
        if not os.path.exists(attendance_file):

            with open(attendance_file, "w") as file:
                file.write("Student ID,Name,Date,Time\n")

        # Check if student already marked today
        try:

            df = pd.read_csv(attendance_file)

            already_marked = (
                (df["Student ID"] == student_id)
                & (df["Date"] == today)
            ).any()

            if already_marked:
                return

        except Exception:
            pass

        # Add attendance
        with open(attendance_file, "a") as file:

            file.write(
                f"{student_id},{name},{today},{current_time}\n"
            )


# =========================
# CAMERA PROCESSOR
# =========================

class FaceRecognitionProcessor(VideoProcessorBase):

    def recv(self, frame):

        image = frame.to_ndarray(format="bgr24")

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5
        )

        for (x, y, w, h) in faces:

            student_id, confidence = recognizer.predict(
                gray[y:y+h, x:x+w]
            )

            if confidence < 80:

                name = student_names.get(
                    student_id,
                    "Unknown"
                )

                # Draw green rectangle
                cv2.rectangle(
                    image,
                    (x, y),
                    (x+w, y+h),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    image,
                    f"{name} - Present",
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

                # Mark attendance
                mark_attendance(
                    student_id,
                    name
                )

            else:

                cv2.rectangle(
                    image,
                    (x, y),
                    (x+w, y+h),
                    (0, 0, 255),
                    2
                )

                cv2.putText(
                    image,
                    "Unknown",
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )

        return av.VideoFrame.from_ndarray(
            image,
            format="bgr24"
        )


# =========================
# STREAMLIT PAGE
# =========================

st.set_page_config(
    page_title="AI Attendance System",
    page_icon="📷",
    layout="wide"
)

st.title(
    "📷 AI-Driven Automated Student Attendance Management System"
)

st.write(
    "Face Recognition Based Attendance System"
)


# =========================
# SIDEBAR
# =========================

st.sidebar.title("Menu")

option = st.sidebar.radio(
    "Select",
    [
        "Dashboard",
        "Start Attendance",
        "View Attendance"
    ]
)


# =========================
# DASHBOARD
# =========================

if option == "Dashboard":

    st.header("📊 Dashboard")

    if os.path.exists(attendance_file):

        df = pd.read_csv(attendance_file)

        st.metric(
            "Total Attendance Records",
            len(df)
        )

        st.dataframe(
            df,
            use_container_width=True
        )

    else:

        st.info(
            "No attendance records found."
        )


# =========================
# START ATTENDANCE
# =========================

elif option == "Start Attendance":

    st.header("📷 Face Recognition Attendance")

    st.write(
        "Click START below and allow camera permission."
    )

    webrtc_streamer(
        key="attendance",
        video_processor_factory=FaceRecognitionProcessor,

        rtc_configuration={
            "iceServers": [
                {
                    "urls": [
                        "stun:stun.l.google.com:19302"
                    ]
                }
            ]
        },

        media_stream_constraints={
            "video": True,
            "audio": False
        }
    )


# =========================
# VIEW ATTENDANCE
# =========================

elif option == "View Attendance":

    st.header("📋 Attendance Records")

    if os.path.exists(attendance_file):

        df = pd.read_csv(attendance_file)

        st.dataframe(
            df,
            use_container_width=True
        )

    else:

        st.info(
            "No attendance records found."
        )
