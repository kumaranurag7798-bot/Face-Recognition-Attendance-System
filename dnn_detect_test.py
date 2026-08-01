import cv2

# YuNet face detector load karo
detector = cv2.FaceDetectorYN.create(
    "face_detection_yunet.onnx",
    "",
    (320, 320),
    score_threshold=0.6
)

cap = cv2.VideoCapture(0)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
detector.setInputSize((w, h))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    _, faces = detector.detect(frame)

    if faces is not None:
        for face in faces:
            x, y, w_box, h_box = face[0:4].astype(int)
            cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
            cv2.putText(frame, "Face Detected (DL)", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Deep Learning Face Detection - Press q to quit", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()