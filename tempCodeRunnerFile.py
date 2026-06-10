import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime
import pyttsx3  # For text-to-speech (macOS compatible)
from sklearn.neighbors import KNeighborsClassifier  

# Initialize text-to-speech engine (macOS compatible)
engine = pyttsx3.init()

def speak(text):
    engine.say(text)
    engine.runAndWait()

# Initialize video capture and face detection
video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load pre-trained data
try:
    with open('data/names.pkl', 'rb') as w:
        LABELS = pickle.load(w)
    with open('data/faces_data.pkl', 'rb') as f:
        FACES = pickle.load(f)
except FileNotFoundError:
    print("Error: Required data files not found.")
    exit()

# Check and fix data length mismatch
print('Shape of Faces matrix --> ', FACES.shape)
if len(LABELS) != FACES.shape[0]:
    print(f"Data mismatch: FACES has {FACES.shape[0]} samples, but LABELS has {len(LABELS)}.")
    if len(LABELS) < FACES.shape[0]:
        print("Trimming FACES to match LABELS.")
        FACES = FACES[:len(LABELS)]  # Adjust FACES to match the number of LABELS
    else:
        print("Trimming LABELS to match FACES.")
        LABELS = LABELS[:FACES.shape[0]]  # Adjust LABELS to match the number of FACES

# Train KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

# Load background image
if not os.path.exists("background.png"):
    print("Warning: Background image not found. Using blank background.")
    imgBackground = np.zeros((720, 1280, 3), dtype=np.uint8)
else:
    imgBackground = cv2.imread("background.png")

COL_NAMES = ['NAME', 'TIME']

# Attendance folder setup
attendance_folder = "Attendance"
os.makedirs(attendance_folder, exist_ok=True)

while True:
    ret, frame = video.read()
    if not ret:
        print("Error: Unable to read from the camera.")
        break

    # Resize frame to match background size (480x640)
    frame_resized = cv2.resize(frame, (640, 480))

    gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    attendance = None  # Initialize attendance variable
    for (x, y, w, h) in faces:
        crop_img = frame_resized[y:y + h, x:x + w]
        resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)

        try:
            output = knn.predict(resized_img)
        except Exception as e:
            print("Error during prediction:", e)
            continue

        ts = time.time()
        date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
        attendance_file = os.path.join(attendance_folder, f"Attendance_{date}.csv")

        cv2.rectangle(frame_resized, (x, y), (x + w, y + h), (0, 0, 255), 1)
        cv2.rectangle(frame_resized, (x, y - 40), (x + w, y), (50, 50, 255), -1)
        cv2.putText(frame_resized, str(output[0]), (x, y - 15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
        attendance = [str(output[0]), str(timestamp)]

    if imgBackground is not None:
        imgBackground[162:162 + 480, 55:55 + 640] = frame_resized  # Insert resized frame
        cv2.imshow("Frame", imgBackground)

    # Wait for keypress events to take attendance or quit
    k = cv2.waitKey(1)
    if k == ord('o'):  # 'o' key to take attendance
        if attendance:
            exist = os.path.isfile(attendance_file)
            with open(attendance_file, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                if not exist:
                    writer.writerow(COL_NAMES)
                writer.writerow(attendance)
            speak("Attendance taken.")
            print(f"Attendance for {attendance[0]} at {attendance[1]} taken.")
            time.sleep(5)

    if k == ord('q'):  # 'q' key to quit
        break

video.release()
cv2.destroyAllWindows()
