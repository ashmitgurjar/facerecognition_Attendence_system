import cv2
import pickle
import numpy as np
import os
import threading

# Initialize video capture and face detector
video = cv2.VideoCapture(0)

# Check if the camera is accessible
if not video.isOpened():
    print("Error: Camera not accessible. Check permissions or device connection.")
    exit()

# Load Haarcascade
haar_path = 'data/haarcascade_frontalface_default.xml'
if not os.path.exists(haar_path):
    print("Error: Haarcascade file not found. Please ensure it is located in the 'data/' directory.")
    exit()

facedetect = cv2.CascadeClassifier(haar_path)

# Input user name
name = input("Enter Your Name: ").strip()

# List to store resized faces and threading lock
faces_data = []
lock = threading.Lock()

def resize_and_store(crop_img):
    """Resize the face image and store it safely in the list."""
    resized_img = cv2.resize(crop_img, (50, 50))
    with lock:
        if len(faces_data) < 100:  # Ensure max limit of 100 faces
            faces_data.append(resized_img)

# Frame processing
frame_count = 0

print("Press 'q' to quit or wait until 100 faces are captured.")
while True:
    ret, frame = video.read()
    if not ret:
        print("Error: Unable to read from camera.")
        break

    frame_count += 1
    if frame_count % 3 != 0:
        continue  # Process every 3rd frame

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    threads = []
    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w]
        thread = threading.Thread(target=resize_and_store, args=(crop_img,))
        threads.append(thread)
        thread.start()

        # Draw rectangle and show progress
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)
        cv2.putText(frame, f"Captured: {len(faces_data)}/100", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 255), 2)

    for thread in threads:
        thread.join()

    cv2.imshow("Capturing Faces", frame)

    if cv2.waitKey(1) == ord('q') or len(faces_data) >= 100:
        break

# Cleanup
video.release()
cv2.destroyAllWindows()

# Exit if no faces captured
if len(faces_data) == 0:
    print("No faces were captured. Exiting without saving.")
    exit()

# Convert to NumPy array and reshape
faces_data = np.asarray(faces_data, dtype=np.uint8)
faces_data = faces_data.reshape(faces_data.shape[0], -1)  # Flatten images

# Ensure data directory exists
if not os.path.exists('data'):
    os.makedirs('data')

# Save or update names.pkl
names_file = 'data/names.pkl'
if not os.path.exists(names_file):
    names = [name] * len(faces_data)
    with open(names_file, 'wb') as f:
        pickle.dump(names, f)
else:
    with open(names_file, 'rb') as f:
        names = pickle.load(f)
    names.extend([name] * len(faces_data))
    with open(names_file, 'wb') as f:
        pickle.dump(names, f)

# Save or update faces_data.pkl
faces_file = 'data/faces_data.pkl'
if not os.path.exists(faces_file):
    with open(faces_file, 'wb') as f:
        pickle.dump(faces_data, f)
else:
    with open(faces_file, 'rb') as f:
        faces = pickle.load(f)
    if faces.shape[1] != faces_data.shape[1]:
        print(f"Error: Face data dimensions mismatch. Existing: {faces.shape[1]}, New: {faces_data.shape[1]}")
        print("Resolution options:")
        print("1. Delete 'data/faces_data.pkl' to start fresh.")
        print("2. Align new data processing to match existing data.")
        exit()
    faces = np.vstack((faces, faces_data))
    with open(faces_file, 'wb') as f:
        pickle.dump(faces, f)

print("Face data saved successfully!")
