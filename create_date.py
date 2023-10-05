import cv2
import os

haar_file = 'haarcascade_frontalface_default.xml'
datasets = 'fotos'

# Initialize face cascade classifier
face_cascade = cv2.CascadeClassifier(haar_file)
webcam = cv2.VideoCapture(0)
count = -1
while count < 100:
    count+=1
    (_, im) = webcam.read()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 4)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (130, 100))

        # Recognize the face here and get the person's name
        # Replace the following line with your face recognition code
        # For now, we'll use a placeholder name "PersonX"
        person_name = "ColoqueSeuNomeAqui"

        # Create a folder for each recognized face
        person_folder = os.path.join(datasets, person_name)
        if not os.path.isdir(person_folder):
            os.mkdir(person_folder)

        # Count existing images in the folder
        image_count = len(os.listdir(person_folder))

        # Save the face as an image in the person's folder
        image_path = os.path.join(person_folder, f"{image_count + 1}.png")
        cv2.imwrite(image_path, face_resize)

    cv2.imshow('OpenCV', im)
    key = cv2.waitKey(10)

    if key == 27:
        break

webcam.release()
cv2.destroyAllWindows()
