import cv2
import numpy as np
import face_recognition
import os

pathToImage = '.venv/ImageAttendance'
listOfImages = os.listdir(pathToImage)
print(listOfImages)

path = '.venv/ImageAttendance'
images = []
classNames = []
studentNames = []
myList = os.listdir(path)
print(myList)

for studentImg in myList:
    curImg = cv2.imread(f'{path}/{studentImg}')
    images.append(curImg)
    studentNames.append(os.path.splitext(studentImg)[0])

print(classNames)

encodeList = []

for img in images:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    encoding = face_recognition.face_encodings(img)
    encodeList.append(encoding)

print(studentNames)
print(len(encodeList))

cap = cv2.VideoCapture(0)

while True:
    succes, img = cap.read()
    imgSmall = cv2.resize(img, (0,0), None, 0.25, 0.25)
    imgSmall = cv2.cvtColor(imgSmall, cv2.COLOR_BGR2RGB)

    currentFaceLocation = face_recognition.face_locations(img)[0]
    currentFaceEncoding = face_recognition.face_encodings(img)[0]

    matches = face_recognition.compare_faces(encodeList, currentFaceEncoding)
    distance = face_recognition.face_distance(encodeList, currentFaceEncoding)
    print(matches)
    print(distance)

    cv2.imshow('Webcam', img)
    cv2.waitKey(1)

