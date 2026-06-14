import cv2 as cv
import numpy as np
import face_recognition
import time
import os

#import serial
#import time

#aud= serial.Serial('COM7', 115200)

CONFIDENCE_THRESHOLD = 0.4
NMS_THRESHOLD = 0.3

COLORS = [(255,0,0),(255,0,255),(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
GREEN =(0,255,0)
BLACK =(0,0,0)
FONTS = cv.FONT_HERSHEY_COMPLEX

class_names = []
with open("classes.txt", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]

yoloNet = cv.dnn.readNet('yolov4-tiny.weights', 'yolov4-tiny.cfg')

yoloNet.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
yoloNet.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)

model = cv.dnn_DetectionModel(yoloNet)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

path = 'dataset'
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

def findEncodings(images):
    encodeList =[]
    for img in images:
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodeListKnown = findEncodings(images)

def object_detector(image):
    classes, scores, boxes = model.detect(image, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    data_list = []

    for (classid, score, box) in zip(classes, scores, boxes):
        class_name = class_names[classid[0]]
        color = COLORS[int(classid) % len(COLORS)]

        cv.rectangle(image, box, color, 2)
        cv.putText(image, class_name, (box[0], box[1]-14), FONTS, 0.5, color, 2)

        data_list.append([class_name, box[2], (box[0], box[1]-2)])
        #aud.write(b'A')
        if class_name == "chair":
            print("chair detected")
            #aud.write(b'B')
            
        elif class_name == "book":
            print("book detected")
            #aud.write(b'C')

        elif class_name == "clock":
            print("clock detected")
            #aud.write(b'D')

        elif class_name == "diningtable":
            print("REMOTE detected")
            #aud.write(b'E')

        elif class_name == "person":
            print("person detected")
            #aud.write(b'F')

        elif class_name == "motorbike":
            print("motorbike detected")
            #aud.write(b'G')

        elif class_name == "car":
            print("car detected")
            #aud.write(b'A')

        elif class_name == "bench":
            print("bench detected")
            #aud.write(b'H')
        

        elif class_name == "handbag":
            print("handbag detected")
            
        elif class_name == "umbrella":
            print("umbrella detected")
            #aud.write(b'I')

        elif class_name == "bottle":
            print("bottle detected")
            #aud.write(b'J')

        elif class_name == "laptop":
            print("laptop detected")
            #aud.write(b'K')

        elif class_name == "keyboard":
            print("keyboard detected")

        elif class_name == "cell phone":
            print("cell phone detected")

        else:
            pass

    return data_list


cap = cv.VideoCapture(0)
counts = 0

while True:
    ret, frame = cap.read()
    imgS = cv.resize(frame, (0, 0), None, 0.25, 0.25)
    imgS = cv.cvtColor(imgS, cv.COLOR_BGR2RGB)
   
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    data = object_detector(frame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4

            cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv.rectangle(frame, (x1, y2-35), (x2, y2), (0, 250, 0), cv.FILLED)

            if name in ['BRUNTHA1', 'BRUNTHA2']:
                cv.putText(frame, "BRUNTHA", (x1+6, y2-6),
                           cv.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                print("BRUNTHA HERE")
            
    cv.imshow('Blind People Helping Application', frame)

    key = cv.waitKey(1)
    if key == ord('q'):
        break

cv.destroyAllWindows()
cap.release()
