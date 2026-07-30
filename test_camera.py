import cv2

# Webcam start karo
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera open nahi ho pa raha!")
else:
    print("Camera successfully khula!")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    cv2.imshow('Test Camera', frame)
    
    # 'q' dabane se band ho jayega
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()