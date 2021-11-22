import cv2
import face_recognition
import numpy as np

imgElon = face_recognition.load_image_file('demoassets/el1.jpg')
imgElon = cv2.cvtColor(imgElon, cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file('demoassets/el3.jpg') #for match
# imgTest = face_recognition.load_image_file('demoassets/RDJ.jpg') #for no match
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

faceloc = face_recognition.face_locations(imgElon)[0]
encodeElon = face_recognition.face_encodings(imgElon)[0]
cv2.rectangle(imgElon,(faceloc[3],faceloc[0]),(faceloc[1],faceloc[2]),(255,0,255),2)

faceloctest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(faceloctest[3],faceloctest[0]),(faceloctest[1],faceloctest[2]),(255,0,255),2)

results = face_recognition.compare_faces([encodeElon],encodeTest)
facedist = face_recognition.face_distance([encodeTest],encodeElon)
print(results,facedist)

cv2.putText(imgTest,f'{results}{round(facedist[0],2)}',(50,40),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,0,255),2)
cv2.imshow('Elon Musk', imgElon)
cv2.imshow('Test', imgTest)
cv2.waitKey(0)