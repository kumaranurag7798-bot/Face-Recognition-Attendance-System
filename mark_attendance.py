import cv2
import pickle
import csv
import os
from datetime import datetime

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')

# Label names load karo
with open('trainer/labels.pickle', 'rb') as f:
    label_names = pickle.load(f)

# Attendance CSV file setup
attendance_file = 'attendance/attendance.csv'
if not os.path.exists('attendance'):
    os.makedirs('attendance')

if not os.path.exists(attendance_file):
    with open(attendance_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Name', 'Date', 'Time'])

# Aaj kisska attendance already mark ho chuka hai, track karne ke liye
marked_today = set()
today_date = datetime.now().strftime('%Y-%m-%d')

# Existing entries check karo (agar already mark hai to dobara na ho)
with open(attendance_file, 'r') as f:
    reader = csv.reader(f)
    next(reader, None)  # header skip karo
    for row in reader:
        if len(row) >= 2 and row[1] == today_date:
            marked_today.add(row[0])

cap = cv2.VideoCapture(0)
print("Camera chalu ho gaya. 'q' dabake band karo.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face_img = gray[y:y+h, x:x+w]
        label, confidence = recognizer.predict(face_img)

        # Kam confidence value = zyada accurate match (LBPH mein aisa hi hota hai)
        if confidence < 70:
            name = label_names.get(label, "Unknown")
            color = (0, 255, 0)

            if name not in marked_today:
                now = datetime.now()
                with open(attendance_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([name, now.strftime('%Y-%m-%d'), now.strftime('%H:%M:%S')])
                marked_today.add(name)
                print(f"Attendance marked for {name}")
        else:
            name = "Unknown"
            color = (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow('Attendance System - Press q to quit', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("System band ho gaya. Attendance CSV check karo: attendance/attendance.csv")