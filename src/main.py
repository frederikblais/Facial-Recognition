import cv2
import face_recognition

imgElon = face_recognition.load_image_file('ImageBasic/musk1.jpg')
imgElon = cv2.cvtColor(imgElon, cv2.COLOR_BGR2RGB)
imgCook = face_recognition.load_image_file('ImageBasic/cook1.jpg')
imgCook = cv2.cvtColor(imgCook, cv2.COLOR_BGR2RGB)

faceLocationElon = face_recognition.face_locations(imgElon)[0]
faceLocationCook = face_recognition.face_locations(imgCook)[0]

faceEncodeElon = face_recognition.face_encodings(imgElon)
faceEncodeCook = face_recognition.face_encodings(imgCook)

imgWho = face_recognition.load_image_file('ImageBasic/whoThis1.jpeg')
imgWho = cv2.cvtColor(imgWho, cv2.COLOR_BGR2RGB)
faceLocationWho = face_recognition.face_locations(imgWho)[0]
faceEncodeWho = face_recognition.face_encodings(imgWho)

# Compare new images
compareToCook = face_recognition.compare_faces(faceEncodeCook, faceEncodeWho)
compareToElon = face_recognition.compare_faces(faceEncodeElon, faceEncodeWho)

cv2.rectangle(imgElon, (faceLocationElon[3], faceLocationElon[0]), (faceLocationElon[1], faceLocationElon[2]), (255, 0, 255), 2)

cv2.imshow('Img Elon', imgElon)

cv2.waitKey(0)