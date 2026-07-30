import cv2
import os

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

# Person ka naam poocho
person_name = input("Person ka naam likho: ")

# Folder banao us person ke liye
save_path = f"dataset/{person_name}"
if not os.path.exists(save_path):
    os.makedirs(save_path)

count = 0
print("Camera ke saamne dekho... 30 photos capture hongi automatically")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        count += 1
        # Sirf face wala part crop karke save karo
        face_img = gray[y:y+h, x:x+w]
        cv2.imwrite(f"{save_path}/{count}.jpg", face_img)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, f"Captured: {count}/30", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow('Capturing Faces - Press q to stop', frame)

    if cv2.waitKey(1) & 0xFF == ord('q') or count >= 30:
        break

cap.release()
cv2.destroyAllWindows()
print(f"Total {count} photos saved for {person_name}!")