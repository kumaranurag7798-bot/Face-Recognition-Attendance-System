# 🎯 Face Recognition Based Attendance System

[![Python](https://img.shields.io/badge/Python-3.14-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5.0.0-green.svg)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-success.svg)]()

A real-time **Face Recognition Attendance System** built with Python and OpenCV. It detects and recognizes faces through a webcam and automatically logs attendance (Name, Date, Time) into a CSV file — eliminating manual roll-calls and proxy attendance.

---

## 📸 Demo

> Add a screenshot or GIF of the system in action here.
> Example: `![Demo](assets/demo.gif)`

| Face Detection | Attendance Marked |
|---|---|
| _screenshot here_ | _screenshot here_ |

---

## ✨ Features

- 🎥 **Real-time face detection** using Haar Cascade Classifier
- 🧠 **Face recognition** using LBPH (Local Binary Pattern Histogram) algorithm
- 📝 **Automatic attendance logging** into a CSV file with Name, Date, and Time
- 🚫 **Duplicate prevention** — each person is marked present only once per day
- 📂 **Easy dataset management** — add new people by just running a capture script
- ⚡ **Lightweight** — no external cloud dependency, runs fully offline

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3 |
| Computer Vision | OpenCV (opencv-contrib-python) |
| Face Detection | Haar Cascade Classifier |
| Face Recognition | LBPH Face Recognizer |
| Data Handling | NumPy, Pandas, CSV |
| Model Storage | YAML + Pickle |

---

## 📁 Project Structure

```
Face-Recognition-Attendance-System/
│
├── dataset/                 # Captured face images (per person, gitignored)
├── trainer/                 # Trained model files (gitignored)
├── attendance/               # Attendance CSV records
│
├── capture_faces.py          # Step 1: Capture face dataset for a new person
├── train_model.py            # Step 2: Train the LBPH recognizer on dataset
├── mark_attendance.py        # Step 3: Run live recognition & mark attendance
│
├── haarcascade_frontalface_default.xml   # Pre-trained face detector
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/kumaranurag7798-bot/face-recognition-attendance-system.git
   cd face-recognition-attendance-system
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

### Step 1: Capture face data for a new person
```bash
python capture_faces.py
```
Enter the person's name when prompted. The webcam will capture 30 face images automatically.

### Step 2: Train the recognition model
```bash
python train_model.py
```
This trains the LBPH recognizer on all captured faces and saves the model to `trainer/trainer.yml`.

### Step 3: Run the attendance system
```bash
python mark_attendance.py
```
The webcam opens, recognizes registered faces in real time, and logs attendance to `attendance/attendance.csv`. Press `q` to quit.

---

## 📊 Sample Output (attendance.csv)

| Name | Date | Time |
|---|---|---|
| Anurag | 2026-07-31 | 09:15:22 |

---

## 🔮 Future Scope

- [ ] Upgrade recognition engine to deep learning-based face embeddings (FaceNet / dlib) for higher accuracy
- [ ] Web-based dashboard (Streamlit/Flask) for live monitoring and analytics
- [ ] Face mask detection module
- [ ] Email/SMS alerts on attendance marking
- [ ] Cloud deployment with multi-camera support

---

## ⚠️ Limitations

- Recognition accuracy depends on lighting conditions
- LBPH is a classical ML approach; deep learning models would offer better accuracy at scale
- Currently designed for a moderate number of users

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
- LinkedIn: [your-linkedin](https://linkedin.com)

---

⭐ If you found this project useful, consider giving it a star!
