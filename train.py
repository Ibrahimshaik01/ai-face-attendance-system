import cv2
import os
import numpy as np

base_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(base_dir, "dataset")
trainer_path = os.path.join(base_dir, "trainer")

os.makedirs(trainer_path, exist_ok=True)

recognizer = cv2.face.LBPHFaceRecognizer_create()

faces = []
ids = []

for folder_name in os.listdir(dataset_path):

    folder_path = os.path.join(dataset_path, folder_name)

    if not os.path.isdir(folder_path):
        continue

    student_id = int(folder_name.split("_")[0])

    for image_name in os.listdir(folder_path):

        image_path = os.path.join(folder_path, image_name)

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            continue

        faces.append(image)
        ids.append(student_id)

print("Training started...")

recognizer.train(faces, np.array(ids))

model_path = os.path.join(trainer_path, "trainer.yml")
recognizer.write(model_path)

print("Training completed successfully!")
print(f"Model saved at: {model_path}")