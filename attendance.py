import cv2
import face_recognition
import numpy as np
import os
from datetime import datetime
# from PIL import ImageGrab

path = 'assets'
images = []
classNames = []
myList = os.listdir(path)
for cls in myList:
    curImg = cv2.imread(f'{path}/{cls}')
    images.append(curImg)
    classNames.append(os.path.splitext(cls)[0])

def findEncoding(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList
def markAttendance(name):
    with open('attendanceReg.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            tmString = now.strftime('%H:%M:%S')
            dayString = now.strftime('%A')
            dtString = now.strftime('%d-%m-%y')
            f.writelines(f'\n{name},{tmString},{dtString},{dayString}')

#### FOR CAPTURING SCREEN RATHER THAN WEBCAM
# def captureScreen(bbox=(300,300,690+300,530+300)):
#     capScr = np.array(ImageGrab.grab(bbox))
#     capScr = cv2.cvtColor(capScr, cv2.COLOR_RGB2BGR)
#     return capScr

encodeListKnown = findEncoding(images)
print(len(encodeListKnown))

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    # img = captureScreen()
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    facesCurrent = face_recognition.face_locations(imgS)
    encodeCurrent = face_recognition.face_encodings(imgS,facesCurrent)

    for encodeFace,faceloc in zip(encodeCurrent,facesCurrent):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
        # print(faceDis)
        matchIndex = np.argmin(faceDis)

        # if matches[matchIndex]:
        #     name = classNames[matchIndex].upper()
        #     # print(name)
        #     y1, x2, y2, x1 = faceloc
        #     y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
        #     cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        #     cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
        #     cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
        #     markAttendance(name)
        if faceDis[matchIndex] < 0.50:
            name = classNames[matchIndex].upper()
            markAttendance(name)
        else:
            name = 'Unknown'
        # print(name)
        y1, x2, y2, x1 = faceloc
        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)


    cv2.imshow('Webcam',img)
    cv2.waitKey(1)