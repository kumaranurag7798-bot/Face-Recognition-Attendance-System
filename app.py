import streamlit as st
import cv2
import os
import numpy as np
import pandas as pd
from datetime import datetime
from collections import deque
from io import BytesIO

st.set_page_config(page_title="Face Recognition Attendance System", page_icon="🎯")

st.title("🎯 Face Recognition Attendance System")
st.caption("AI-powered attendance system using Python, OpenCV and Deep Learning (YuNet + SFace)")

# Load Deep Learning models once
detector = cv2.FaceDetectorYN.create("face_detection_yunet.onnx", "", (320, 320), score_threshold=0.6)
recognizer = cv2.FaceRecognizerSF.create("face_recognition_sface.onnx", "")

MATCH_THRESHOLD = 0.75

# ---------------------------------------------------
# DASHBOARD (Quick Stats)
# ---------------------------------------------------
st.divider()

total_registered = 0
if os.path.exists("embeddings"):
    total_registered = len([f for f in os.listdir("embeddings") if f.endswith(".npy")])

present_today = 0
today_date_str = datetime.now().strftime('%Y-%m-%d')
if os.path.exists("attendance/attendance.csv"):
    att_df = pd.read_csv("attendance/attendance.csv")
    present_today = len(att_df[att_df["Date"] == today_date_str]["Name"].unique())

attendance_percentage = 0
if total_registered > 0:
    attendance_percentage = round((present_today / total_registered) * 100, 1)

col1, col2, col3 = st.columns(3)
col1.metric("👥 Total Registered Users", total_registered)
col2.metric("✅ Present Today", present_today)
col3.metric("📊 Attendance %", f"{attendance_percentage}%")

# ---------------------------------------------------
# SECTION 1: REGISTER NEW PERSON (Deep Learning - SFace)
# ---------------------------------------------------
st.divider()
st.subheader("📸 Register New Person (Deep Learning)")

person_name = st.text_input("Enter person's name")
start_capture = st.button("Start Capturing (15 frames)")

if start_capture:
    if person_name.strip() == "":
        st.warning("Pehle naam daalo!")
    else:
        os.makedirs("embeddings", exist_ok=True)

        cap = cv2.VideoCapture(0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        detector.setInputSize((w, h))

        frame_placeholder = st.empty()
        progress_bar = st.progress(0)
        collected_features = []

        while len(collected_features) < 15:
            ret, frame = cap.read()
            if not ret:
                break

            _, faces = detector.detect(frame)

            if faces is not None and len(faces) > 0:
                # Sirf sabse bada face lo (registration ke waqt bhi false detections se bachne ke liye)
                face = max(faces, key=lambda f: f[2] * f[3])
                x, y, w_box, h_box = face[0:4].astype(int)

                aligned_face = recognizer.alignCrop(frame, face)
                feature = recognizer.feature(aligned_face)
                collected_features.append(feature)

                cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
                cv2.putText(frame, f"{len(collected_features)}/15", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            frame_placeholder.image(frame, channels="BGR")
            progress_bar.progress(min(len(collected_features) / 15, 1.0))

        cap.release()

        avg_embedding = np.mean(collected_features, axis=0)
        np.save(f"embeddings/{person_name}.npy", avg_embedding)

        st.success(f"✅ {person_name} registered successfully with Deep Learning embeddings!")

# ---------------------------------------------------
# SECTION 2: INFO (No training needed with Deep Learning)
# ---------------------------------------------------
st.divider()
st.info("ℹ️ Deep Learning approach mein alag se 'training' ki zarurat nahi — har naya person register karte hi uska embedding ready ho jata hai!")

# ---------------------------------------------------
# SECTION 3: MARK ATTENDANCE (Deep Learning - SFace + Rolling Average, single-face fix)
# ---------------------------------------------------
st.divider()
st.subheader("✅ Mark Attendance (Deep Learning)")

if not os.path.exists("embeddings") or len(os.listdir("embeddings")) == 0:
    st.warning("Pehle upar se kam se kam ek person register karo.")
else:
    duration = st.slider("Kitni der camera chalu rahe (seconds)", 5, 30, 15)
    start_attendance = st.button("Start Recognition")

    if start_attendance:
        # Saare saved embeddings load karo
        known_embeddings = {}
        for file_name in os.listdir("embeddings"):
            if file_name.endswith(".npy"):
                p_name = file_name.replace(".npy", "")
                known_embeddings[p_name] = np.load(f"embeddings/{file_name}")

        # Har person ke liye last 5 frames ke score store karne ke liye
        score_history = {p_name: deque(maxlen=5) for p_name in known_embeddings.keys()}

        os.makedirs('attendance', exist_ok=True)
        attendance_file = 'attendance/attendance.csv'
        if not os.path.exists(attendance_file):
            pd.DataFrame(columns=['Name', 'Date', 'Time']).to_csv(attendance_file, index=False)

        today_date = datetime.now().strftime('%Y-%m-%d')
        existing_df = pd.read_csv(attendance_file)
        marked_today = set(existing_df[existing_df['Date'] == today_date]['Name'])

        cap = cv2.VideoCapture(0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        detector.setInputSize((w, h))

        frame_placeholder = st.empty()
        status_placeholder = st.empty()
        start_time = datetime.now()
        new_entries = []

        while (datetime.now() - start_time).seconds < duration:
            ret, frame = cap.read()
            if not ret:
                break

            _, faces = detector.detect(frame)

            if faces is not None and len(faces) > 0:
                # Sirf sabse bada detected face process karo (real face),
                # taaki chhoti galat detections (products, wallpaper, etc.) score history mein mix na ho
                face = max(faces, key=lambda f: f[2] * f[3])
                x, y, w_box, h_box = face[0:4].astype(int)

                aligned_face = recognizer.alignCrop(frame, face)
                current_feature = recognizer.feature(aligned_face)

                # Is frame mein har known person ke saath score nikalo aur history mein daalo
                for p_name, ref_embedding in known_embeddings.items():
                    score = recognizer.match(ref_embedding, current_feature, cv2.FaceRecognizerSF_FR_COSINE)
                    score_history[p_name].append(score)

                # Ab har person ka rolling average nikalo (jitne frames abhi tak collect hue)
                best_match_name = "Unknown"
                best_avg_score = 0

                for p_name, history in score_history.items():
                    if len(history) == 0:
                        continue
                    avg_score = sum(history) / len(history)
                    if avg_score > best_avg_score:
                        best_avg_score = avg_score
                        best_match_name = p_name

                if best_avg_score >= MATCH_THRESHOLD:
                    name = best_match_name
                    color = (0, 255, 0)
                    if name not in marked_today:
                        now = datetime.now()
                        new_entries.append([name, now.strftime('%Y-%m-%d'), now.strftime('%H:%M:%S')])
                        marked_today.add(name)
                        status_placeholder.success(f"Attendance marked: {name} (avg score: {best_avg_score:.2f})")
                else:
                    name = "Unknown"
                    color = (0, 0, 255)

                cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), color, 2)
                cv2.putText(frame, f"{name} ({best_avg_score:.2f})", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            frame_placeholder.image(frame, channels="BGR")

        cap.release()

        if new_entries:
            new_df = pd.DataFrame(new_entries, columns=['Name', 'Date', 'Time'])
            new_df.to_csv(attendance_file, mode='a', header=False, index=False)

        st.info("Session khatam ho gaya.")

# ---------------------------------------------------
# SECTION 4: SHOW ATTENDANCE RECORDS (with Search + Date Filter)
# ---------------------------------------------------
st.divider()
st.subheader("📋 Attendance Records")

if os.path.exists('attendance/attendance.csv'):
    df = pd.read_csv('attendance/attendance.csv')

    # ---- Search & Date Filter ----
    filter_col1, filter_col2 = st.columns(2)

    with filter_col1:
        search_name = st.text_input("🔍 Search by name")

    with filter_col2:
        unique_dates = ["All Dates"] + sorted(df["Date"].unique().tolist(), reverse=True)
        selected_date = st.selectbox("📅 Filter by date", unique_dates)

    filtered_df = df.copy()

    if search_name.strip() != "":
        filtered_df = filtered_df[filtered_df["Name"].str.contains(search_name, case=False, na=False)]

    if selected_date != "All Dates":
        filtered_df = filtered_df[filtered_df["Date"] == selected_date]

    df = filtered_df  # ab niche CSV/Excel download bhi isी filtered data ka hoga
    # ---- End Filter ----

    st.dataframe(df, use_container_width=True)

    # CSV download (filtered data)
    st.download_button(
        "Download CSV",
        df.to_csv(index=False),
        "attendance.csv",
        "text/csv"
    )

    # Excel (.xlsx) download (filtered data)
    excel_buffer = BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Attendance")
    excel_buffer.seek(0)

    st.download_button(
        "Download Excel (.xlsx)",
        excel_buffer,
        "attendance.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
else:
    st.write("Abhi tak koi attendance record nahi hai.")