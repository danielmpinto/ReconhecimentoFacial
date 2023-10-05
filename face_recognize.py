import cv2
import os
import numpy as np

size = 4
haar_file = 'haarcascade_frontalface_default.xml'
datasets = 'fotos'

# Create a list of names from subdirectories in 'fotos' folder
names = [name for name in os.listdir(datasets) if os.path.isdir(os.path.join(datasets, name))]

# Initialize the face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + haar_file)

# Define the width and height for resizing
width, height = 130, 100

# Create a LBPH Face Recognizer
model = cv2.face_LBPHFaceRecognizer.create()

# Create a list to store training data
training_data = []

# Load and preprocess training images
for name in names:
    folder_path = os.path.join(datasets, name)
    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_name)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (width, height))
        training_data.append((image, names.index(name)))

# Train the model
labels = [label for (_, label) in training_data]
images = [image for (image, _) in training_data]
model.train(images, np.array(labels))

# Save the trained model to a file
model.save('trained_model.xml')

# Load the trained model
model.read('trained_model.xml')

# Open the webcam
webcam = cv2.VideoCapture(0)

while True:
    _, frame = webcam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (width, height))

        # Try to recognize the face
        prediction = model.predict(face_resize)

        if prediction[1] < 500:
            name = names[prediction[0]]
            confidence = int(100 * (1 - (prediction[1]) / 300))
            if confidence > 75:
                cv2.putText(frame, name, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, str(confidence) + '%', (x, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, 'Quem es tu?', (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(frame, 'Nao te conheço!', (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('Detectando Faces', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()
