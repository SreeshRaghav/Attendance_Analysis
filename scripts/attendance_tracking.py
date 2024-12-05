import cv2
import pickle
import pandas as pd
from datetime import datetime
import os

# Paths for the model, labels, and logs
MODEL_PATH = 'models/face_recognizer.yml'
LABELS_PATH = 'models/labels.pkl'
LOG_PATH = 'logs/attendance.csv'

# Load the trained model and label mappings
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(MODEL_PATH)

with open(LABELS_PATH, 'rb') as f:
    label_map = pickle.load(f)

# Create attendance DataFrame if it doesn't exist
if not os.path.exists(LOG_PATH):
    attendance = pd.DataFrame(columns=['Name', 'Time'])
else:
    attendance = pd.read_csv(LOG_PATH)

# Start video capture
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

    for (x, y, w, h) in faces:
        roi_gray = gray_frame[y:y+h, x:x+w]
        label, confidence = recognizer.predict(roi_gray)

        if confidence < 50:  # Adjust confidence threshold as needed
            name = label_map[label]
            now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            # Log attendance using pd.concat
            new_row = pd.DataFrame({'Name': [name], 'Time': [now]})
            attendance = pd.concat([attendance, new_row], ignore_index=True)

            # Draw rectangle and name on the frame
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    # Display the video feed
    cv2.imshow('Face Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Save the attendance log to CSV
attendance.to_csv(LOG_PATH, index=False)

cap.release()
cv2.destroyAllWindows()
