import cv2
import numpy as np

detector = cv2.FaceDetectorYN.create(
    "face_detection_yunet.onnx",
    "",
    (320, 320),
    score_threshold=0.6
)

recognizer = cv2.FaceRecognizerSF.create(
    "face_recognition_sface.onnx",
    ""
)

cap = cv2.VideoCapture(0)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
detector.setInputSize((w, h))

reference_feature = None
collected_features = []
collecting = False

print("Camera khula hai.")
print("'s' dabao — 15 frames collect karega tera face ke, fir average nikalega.")
print("'q' dabake band karo.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    _, faces = detector.detect(frame)
    current_feature = None

    if faces is not None and len(faces) > 0:
        face = faces[0]
        x, y, w_box, h_box = face[0:4].astype(int)

        aligned_face = recognizer.alignCrop(frame, face)
        current_feature = recognizer.feature(aligned_face)

        cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (255, 0, 0), 2)

        if collecting:
            collected_features.append(current_feature)
            cv2.putText(frame, f"Collecting: {len(collected_features)}/15", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            if len(collected_features) >= 15:
                reference_feature = np.mean(collected_features, axis=0)
                collecting = False
                collected_features = []
                print("✅ Reference saved (averaged from 15 frames)!")

        elif reference_feature is not None:
            score = recognizer.match(reference_feature, current_feature, cv2.FaceRecognizerSF_FR_COSINE)
            print(f"Raw Cosine Score: {score:.4f}")

            label = f"Score: {score:.2f}"
            color = (0, 255, 0) if score > 0.363 else (0, 0, 255)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        else:
            cv2.putText(frame, "Press 's' to register", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imshow("SFace Recognition Test - s=save, q=quit", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        collecting = True
        collected_features = []
        print("📸 Collecting frames... apna face steady rakho camera ke saamne.")
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()