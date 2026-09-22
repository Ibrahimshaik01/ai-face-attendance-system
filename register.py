import cv2
import os

student_id = input("Enter Student ID: ")
student_name = input("Enter Student Name: ")
base_dir = os.path.dirname(os.path.abspath(__file__))

dataset_folder = os.path.join(base_dir, "dataset")
os.makedirs(dataset_folder, exist_ok=True)

folder = os.path.join(dataset_folder, f"{student_id}_{student_name}")
os.makedirs(folder, exist_ok=True)

camera = cv2.VideoCapture(0)

face_detector = cv2.CascadeClassifier(
    "models/haarcascade_frontalface_default.xml"
)

count = 0

print("Camera started...")
print("Look at the camera. Press Q to stop.")

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
        count += 1

        face = gray[y:y+h, x:x+w]

        filename = f"{folder}/{count}.jpg"
        cv2.imwrite(filename, face)

        cv2.rectangle(
            frame,
            (x, y),
            (x+w, y+h),
            (255, 0, 0),
            2
        )

        cv2.putText(
            frame,
            f"Samples: {count}/50",
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2
        )

    cv2.imshow("Student Registration", frame)

    if cv2.waitKey(100) & 0xFF == ord("q"):
        break

    if count >= 50:
        break

camera.release()
cv2.destroyAllWindows()

print(f"\nRegistration completed for {student_name}!")
print(f"Saved {count} face samples.")