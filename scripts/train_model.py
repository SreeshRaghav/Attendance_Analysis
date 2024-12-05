import os
import cv2
import pickle

DATASET_PATH = 'datasets'
MODEL_PATH = 'models/face_recognizer.yml'
LABELS_PATH = 'models/labels.pkl'

# Initialize OpenCV LBPH face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Prepare training data
faces = []
labels = []
label_map = {}
current_id = 0

for person_name in os.listdir(DATASET_PATH):
    person_path = os.path.join(DATASET_PATH, person_name)
    label_map[current_id] = person_name

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        gray_image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        faces.append(gray_image)
        labels.append(current_id)

    current_id += 1

# Train the recognizer
recognizer.train(faces, labels)

# Save the trained model and label mappings
recognizer.save(MODEL_PATH)
with open(LABELS_PATH, 'wb') as f:
    pickle.dump(label_map, f)

print("Training completed and model saved!")
