import cv2
import os
from datetime import datetime

base_dir = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(base_dir, "trainer", "trainer.yml")
cascade_path = os.path.join(
    base_dir,
    "models",
    "haarcascade_frontalface_default.xml"
)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(model_path)

face_detector = cv2.CascadeClassifier(cascade_path)

student_names = {
    101: "Ibrahim",
    102: "Tanveer",
    103: "Baba",
    104: "mujju"
}

attendance_file = os.path.join(base_dir, "attendance", "attendance.csv")

os.makedirs(os.path.dirname(attendance_file), exist_ok=True)

if not os.path.exists(attendance_file):
    with open(attendance_file, "w") as file:
        file.write("Student ID,Name,Date,Time\n")

camera = cv2.VideoCapture(0)

print("Attendance system started...")
print("Press Q to stop.")

marked_today = set()

while True:

    ret, frame = camera.read()

    if not ret:
        print("Could not access camera.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:

        student_id, confidence = recognizer.predict(
            gray[y:y+h, x:x+w]
        )
        print(f"ID: {student_id}, Confidence: {confidence:.2f}")
        if confidence < 80:

            name = student_names.get(student_id, "Unknown")

            cv2.putText(
                frame,
                f"{name} - Present",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.rectangle(
                frame,
                (x, y),
                (x+w, y+h),
                (0, 255, 0),
                2
            )

            today = datetime.now().strftime("%Y-%m-%d")

            if student_id not in marked_today:

                current_time = datetime.now().strftime("%H:%M:%S")

                with open(attendance_file, "a") as file:
                    file.write(
                        f"{student_id},{name},{today},{current_time}\n"
                    )

                marked_today.add(student_id)

                print(f"Attendance marked: {name}")

        else:

            cv2.putText(
                frame,
                "Unknown",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

            cv2.rectangle(
                frame,
                (x, y),
                (x+w, y+h),
                (0, 0, 255),
                2
            )

    cv2.imshow("Face Recognition Attendance", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()

print("Attendance system stopped.")