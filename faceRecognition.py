import cv2 as cv
from cv2 import VideoCapture

haar_cascade = cv.CascadeClassifier('haar_face.xml')

people = ['toukir','shakil','bindu','robin','monika','juli','biva']


face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')


# img = cv.imread(r'demo/temp.jpg')

while True:
    capture = cv.VideoCapture(0)
    success, img = capture.read()


    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Detect the face in the image
    faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 4)

    for (x,y,w,h) in faces_rect:
        faces_roi = gray[y:y+h,x:x+w]

        label, confidence = face_recognizer.predict(faces_roi)
        print(f'Label = {people[label]} with a confidence of {confidence}')

        cv.putText(img, str(people[label]), (20,20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), thickness=2)
        cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=2)

    cv.imshow('Detected Face', img)

    if cv.waitKey(1) & 0xff == ord('q'):
        break