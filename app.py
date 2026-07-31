import streamlit as st
import cv2
import os
import numpy as np
import pandas as pd
import pickle
from datetime import datetime

st.set_page_config(page_title="Face Recognition Attendance System", page_icon="🎯")

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

st.title("🎯 Face Recognition Attendance System")
st.caption("AI-powered attendance system using Python, OpenCV and LBPH")

# ---------------------------------------------------
# SECTION 1: REGISTER NEW PERSON
# ---------------------------------------------------
st.divider()
st.subheader("📸 Register New Person")

person_name = st.text_input("Enter person's name")
start_capture = st.button("Start Capturing 30 Photos")

if start_capture:
    if person_name.strip() == "":
        st.warning("Pehle naam daalo!")
    else:
        save_path = f"dataset/{person_name}"
        os.makedirs(save_path, exist_ok=True)

        cap = cv2.VideoCapture(0)
        frame_placeholder = st.empty()
        progress_bar = st.progress(0)
        count = 0

        while count < 30:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                count += 1
                face_img = gray[y:y+h, x:x+w]
                cv2.imwrite(f"{save_path}/{count}.jpg", face_img)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f"{count}/30", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            frame_placeholder.image(frame, channels="BGR")
            progress_bar.progress(min(count / 30, 1.0))

            if count >= 30:
                break

        cap.release()
        st.success(f"✅ {count} photos captured for {person_name}!")

# ---------------------------------------------------
# SECTION 2: TRAIN MODEL
# ---------------------------------------------------
st.divider()
st.subheader("🧠 Train the Recognition Model")
st.write("Ye saare registered logon ke data pe model train karega.")

if st.button("Train Now"):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    dataset_path = 'dataset'
    faces, labels, label_names = [], [], {}
    current_label = 0

    if not os.path.exists(dataset_path) or len(os.listdir(dataset_path)) == 0:
        st.error("Dataset khali hai! Pehle upar se log register karo.")
    else:
        for person_name in os.listdir(dataset_path):
            person_path = os.path.join(dataset_path, person_name)
            if not os.path.isdir(person_path):
                continue
            label_names[current_label] = person_name
            for image_name in os.listdir(person_path):
                img = cv2.imread(os.path.join(person_path, image_name), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                faces.append(img)
                labels.append(current_label)
            current_label += 1

        recognizer.train(faces, np.array(labels))
        os.makedirs('trainer', exist_ok=True)
        recognizer.save('trainer/trainer.yml')

        with open('trainer/labels.pickle', 'wb') as f:
            pickle.dump(label_names, f)

        st.success(f"✅ Model trained on {len(faces)} images for {len(label_names)} person(s): {', '.join(label_names.values())}")

# ---------------------------------------------------
# SECTION 3: MARK ATTENDANCE
# ---------------------------------------------------
st.divider()
st.subheader("✅ Mark Attendance (Live Recognition)")

if not os.path.exists('trainer/trainer.yml'):
    st.warning("Pehle upar se model train karo.")
else:
    duration = st.slider("Kitni der camera chalu rahe (seconds)", 5, 30, 15)
    start_attendance = st.button("Start Recognition")

    if start_attendance:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read('trainer/trainer.yml')

        with open('trainer/labels.pickle', 'rb') as f:
            label_names = pickle.load(f)

        os.makedirs('attendance', exist_ok=True)
        attendance_file = 'attendance/attendance.csv'
        if not os.path.exists(attendance_file):
            pd.DataFrame(columns=['Name', 'Date', 'Time']).to_csv(attendance_file, index=False)

        today_date = datetime.now().strftime('%Y-%m-%d')
        existing_df = pd.read_csv(attendance_file)
        marked_today = set(existing_df[existing_df['Date'] == today_date]['Name'])

        cap = cv2.VideoCapture(0)
        frame_placeholder = st.empty()
        status_placeholder = st.empty()
        start_time = datetime.now()

        new_entries = []

        while (datetime.now() - start_time).seconds < duration:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                face_img = gray[y:y+h, x:x+w]
                label, confidence = recognizer.predict(face_img)

                if confidence < 70:
                    name = label_names.get(label, "Unknown")
                    color = (0, 255, 0)
                    if name not in marked_today:
                        now = datetime.now()
                        new_entries.append([name, now.strftime('%Y-%m-%d'), now.strftime('%H:%M:%S')])
                        marked_today.add(name)
                        status_placeholder.success(f"Attendance marked: {name}")
                else:
                    name = "Unknown"
                    color = (0, 0, 255)

                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            frame_placeholder.image(frame, channels="BGR")

        cap.release()

        if new_entries:
            new_df = pd.DataFrame(new_entries, columns=['Name', 'Date', 'Time'])
            new_df.to_csv(attendance_file, mode='a', header=False, index=False)

        st.info("Session khatam ho gaya.")

# ---------------------------------------------------
# SECTION 4: SHOW ATTENDANCE RECORDS
# ---------------------------------------------------
st.divider()
st.subheader("📋 Attendance Records")

if os.path.exists('attendance/attendance.csv'):
    df = pd.read_csv('attendance/attendance.csv')
    st.dataframe(df, use_container_width=True)
    st.download_button("Download CSV", df.to_csv(index=False), "attendance.csv", "text/csv")
else:
    st.write("Abhi tak koi attendance record nahi hai.")