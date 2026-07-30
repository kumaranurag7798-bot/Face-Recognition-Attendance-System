import cv2
import os
import numpy as np

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

dataset_path = 'dataset'
faces = []
labels = []
label_names = {}
current_label = 0

# Dataset folder ke andar har person ke folder mein jao
for person_name in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person_name)
    
    if not os.path.isdir(person_path):
        continue
    
    label_names[current_label] = person_name
    
    for image_name in os.listdir(person_path):
        image_path = os.path.join(person_path, image_name)
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            continue
        
        faces.append(img)
        labels.append(current_label)
    
    current_label += 1

print(f"Total {len(faces)} images loaded for {len(label_names)} person(s)")

# Model train karo
recognizer.train(faces, np.array(labels))

# Model save karo
if not os.path.exists('trainer'):
    os.makedirs('trainer')

recognizer.save('trainer/trainer.yml')

# Label names bhi save karo (baad mein naam pata karne ke liye)
import pickle
with open('trainer/labels.pickle', 'wb') as f:
    pickle.dump(label_names, f)

print("Model training complete! trainer/trainer.yml mein save ho gaya")