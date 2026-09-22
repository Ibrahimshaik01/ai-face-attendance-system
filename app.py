import streamlit as st
import cv2
import os
import pandas as pd
from datetime import datetime

st.set_page_config(
    page_title="AI Attendance System",
    page_icon="📷",
    layout="wide"
)

st.title("📷 AI-Driven Automated Student Attendance Management System")
st.write("Face Recognition Based Attendance System")

st.divider()

# Paths
base_dir = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(
    base_dir,
    "trainer",
    "trainer.yml"
)

cascade_path = os.path.join(
    base_dir,
    "models",
    "haarcascade_frontalface_default.xml"
)

attendance_file = os.path.join(
    base_dir,
    "attendance",
    "attendance.csv"
)

# Student names
student_names = {
    101: "Ibrahim",
    102: "Tanveer",
    103: "Baba",
    104: "Revanth"
    # Add the other 3 students here
}

# Load face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(model_path)

face_detector = cv2.CascadeClassifier(cascade_path)

# Sidebar
st.sidebar.title("⚙️ Menu")

option = st.sidebar.radio(
    "Select Option",
    [
        "Dashboard",
        "Start Attendance",
        "View Attendance"
    ]
)

# ---------------- DASHBOARD ----------------

if option == "Dashboard":

    st.header("📊 Dashboard")

    if os.path.exists(attendance_file):

        df = pd.read_csv(attendance_file)

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Attendance Records",
                len(df)
            )

        with col2:
            st.metric(
                "Students Registered",
                len(student_names)
            )

        st.subheader("Recent Attendance")

        st.dataframe(
            df,
            width="stretch"
        )

    else:
        st.info("No attendance records found.")

# ---------------- START ATTENDANCE ----------------

elif option == "Start Attendance":

    st.header("📷 Face Recognition Attendance")

    start = st.button(
        "Start Camera",
        type="primary"
    )

    stop = st.button(
        "Stop Camera"
    )

    frame_placeholder = st.empty()

    if start:

        camera = cv2.VideoCapture(0)

        marked_today = set()

        while camera.isOpened():

            ret, frame = camera.read()

            if not ret:
                st.error("Could not access camera.")
                break

            gray = cv2.cvtColor(
                frame,
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

                    cv2.rectangle(
                        frame,
                        (x, y),
                        (x+w, y+h),
                        (0, 255, 0),
                        2
                    )

                    cv2.putText(
                        frame,
                        f"{name} - Present",
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2
                    )

                    today = datetime.now().strftime(
                        "%Y-%m-%d"
                    )

                    if student_id not in marked_today:

                        current_time = datetime.now().strftime(
                            "%H:%M:%S"
                        )

                        os.makedirs(
                            os.path.dirname(attendance_file),
                            exist_ok=True
                        )

                        if not os.path.exists(attendance_file):

                            with open(
                                attendance_file,
                                "w"
                            ) as file:

                                file.write(
                                    "Student ID,Name,Date,Time\n"
                                )

                        with open(
                            attendance_file,
                            "a"
                        ) as file:

                            file.write(
                                f"{student_id},{name},{today},{current_time}\n"
                            )

                        marked_today.add(student_id)

                else:

                    cv2.rectangle(
                        frame,
                        (x, y),
                        (x+w, y+h),
                        (0, 0, 255),
                        2
                    )

                    cv2.putText(
                        frame,
                        "Unknown",
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2
                    )

            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            frame_placeholder.image(
                frame,
                channels="RGB"
            )

        camera.release()

# ---------------- VIEW ATTENDANCE ----------------

elif option == "View Attendance":

    st.header("📊 Attendance Records")

    if os.path.exists(attendance_file):

        df = pd.read_csv(attendance_file)

        st.dataframe(
            df,
            width="stretch"
        )

    else:

        st.info(
            "No attendance records available."
        )