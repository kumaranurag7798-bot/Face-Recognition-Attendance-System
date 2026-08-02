# 🎯 Face Recognition Based Attendance System

[![Python](https://img.shields.io/badge/Python-3.14-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5.0.0-green.svg)](https://opencv.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-success.svg)]()

A real-time **AI-powered Face Recognition Attendance System** built with Python, OpenCV, and Deep Learning. It detects and recognizes faces through a webcam using pre-trained neural networks (YuNet + SFace) and automatically logs attendance (Name, Date, Time) into a CSV file — all through an interactive Streamlit web app.

---

## 📸 Demo

> Add a screenshot or GIF of the system in action here.
> Example: `![Demo](assets/demo.gif)`

| Register New Person | Live Attendance Recognition |
|---|---|
| _screenshot here_ | _screenshot here_ |

---

## ✨ Features

- 🎥 **Real-time face detection** using YuNet (deep learning-based detector)
- 🧠 **Face recognition** using SFace — a deep neural network that generates 128-dimensional face embeddings
- 🌐 **Interactive web app** built with Streamlit — register, and mark attendance, all in the browser
- 📊 **Rolling average matching** — averages the last 5 frame scores for stable, reliable recognition
- 📝 **Automatic attendance logging** into a CSV file with Name, Date, and Time
- 🚫 **Duplicate prevention** — each person is marked present only once per day
- 📂 **Simple registration flow** — capture 15 frames, and the person is instantly ready to be recognized (no separate training step needed)
- 📥 **Downloadable attendance records** directly from the web app

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3 |
| Web Framework | Streamlit |
| Computer Vision | OpenCV (opencv-contrib-python) |
| Face Detection | YuNet (Deep Learning, ONNX model) |
| Face Recognition | SFace (Deep Learning face embeddings, ONNX model) |
| Similarity Metric | Cosine Similarity |
| Data Handling | NumPy, Pandas |

---

## 📁 Project Structure

```
Face-Recognition-Attendance-System/
│
├── embeddings/                # Saved face embeddings per person (.npy, gitignored)
├── attendance/                # Attendance CSV records
│
├── app.py                     # Main Streamlit web app (register + attendance)
│
├── face_detection_yunet.onnx      # Pre-trained deep learning face detector
├── face_recognition_sface.onnx    # Pre-trained deep learning face recognizer
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/kumaranurag7798-bot/Face-Recognition-Attendance-System.git
   cd Face-Recognition-Attendance-System
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   venv\Scripts\activate      # Windows
   source venv/bin/activate   # macOS/Linux
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Usage

### Step 1: Run the app
```bash
streamlit run app.py
```
This opens the app in your browser at `localhost:8501`.

### Step 2: Register a new person
Go to the **"Register New Person"** section, enter a name, and click **"Start Capturing (15 frames)"**. The webcam captures 15 frames of the face, generates a deep learning embedding, and saves it under `embeddings/`.

### Step 3: Mark attendance
Go to the **"Mark Attendance"** section, set a duration, and click **"Start Recognition"**. The webcam detects the face, compares it against all registered embeddings using cosine similarity, and marks attendance for the closest match above the confidence threshold.

### Step 4: View records
Scroll down to **"Attendance Records"** to see the live table, and download it as a CSV anytime.

---

## 🧠 How It Works

1. **Detection** — YuNet locates the face in each frame and identifies 5 facial landmarks (eyes, nose, mouth corners).
2. **Alignment & Embedding** — The face is aligned using these landmarks, then SFace converts it into a 128-dimensional numerical embedding — a unique "signature" of that face.
3. **Registration** — 15 embeddings are captured and averaged into one stable reference embedding per person.
4. **Recognition** — For each live frame, a new embedding is generated and compared (via cosine similarity) against all registered embeddings. A rolling average over the last 5 frames smooths out noise from any single bad frame.
5. **Attendance Logging** — If the best match score crosses the threshold, the person's name, date, and time are logged to `attendance.csv` (once per day).

---

## 📊 Sample Output (attendance.csv)

| Name | Date | Time |
|---|---|---|
| Anurag | 2026-08-02 | 17:20:14 |

---

## 🔮 Future Scope

- [ ] Deploy on the cloud with browser-based webcam access (via `streamlit-webrtc`)
- [ ] Add face mask detection module
- [ ] Email/SMS alerts on attendance marking
- [ ] Admin dashboard with attendance analytics and charts
- [ ] Multi-camera support for larger deployments

---

## ⚠️ Limitations

- Recognition accuracy can vary with major changes in lighting, distance, or angle from the registration conditions
- Currently designed for a moderate number of registered users
- Runs locally with direct webcam access (cloud deployment requires additional setup)

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](../../issues).

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Anurag**
- GitHub: [@kumaranurag7798-bot](https://github.com/kumaranurag7798-bot)

---

⭐ If you found this project useful, consider giving it a star!
