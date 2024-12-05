import cv2
import os

# Folder to save images
DATASET_PATH = 'datasets'
if not os.path.exists(DATASET_PATH):
    os.makedirs(DATASET_PATH)

person_name = input("Enter the person's name: ")
person_path = os.path.join(DATASET_PATH, person_name)
os.makedirs(person_path, exist_ok=True)

cap = cv2.VideoCapture(0)
print("Press 'q' to quit capturing images.")
count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Display the frame
    cv2.imshow('Capture Images', frame)

    # Save image
    img_path = os.path.join(person_path, f"{count}.jpg")
    cv2.imwrite(img_path, frame)
    count += 1

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"Collected {count} images for {person_name}.")
