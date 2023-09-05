import cv2
import numpy as np
from roboflow import Roboflow
rf = Roboflow(api_key="Q8frMzPenSHk9xY3093F")
project = rf.workspace("flugunfallerkennung").project("fire-and-smoke-detection-zpsha")
#dataset = project.version(1).download("yolov5")
model = project.version(1).model

rf = Roboflow(api_key="8uqjCkrutuuqGtbdGSus")
project1 = rf.workspace("zinedine-zam").project("initiation")
model1 = project1.version(1).model




# infer on a local image

 #visualize your prediction
#model.predict("your_image.jpg", confidence=40, overlap=30).save("prediction.jpg")

# infer on an image hosted elsewhere
# print(model.predict("URL_OF_YOUR_IMAGE", hosted=True, confidence=40, overlap=30).json())
import glob
import requests
import base64
from base64 import decodebytes
import io
from PIL import Image
import time
import cv2
from io import BytesIO


parts = []
parts1 = []
url_base = 'https://detect.roboflow.com/'
endpoint = 'fire-and-smoke-detection-zpsha'
endpoint1 = 'initiation'
access_token = '?access_token=Q8frMzPenSHk9xY3093F'
access_token1 = '?access_token=8uqjCkrutuuqGtbdGSus'
format = '&format=json'
confidence = '&confidence=10'
stroke='&stroke=5'
parts.append(url_base)
parts.append(endpoint)
parts.append(access_token)
parts.append(format)
parts.append(confidence)
parts.append(stroke)
url = ''.join(parts)
parts1.append(url_base)
parts1.append(endpoint1)
parts1.append(access_token1)
parts1.append(format)
parts1.append(confidence)
parts1.append(stroke)
url1 = ''.join(parts)

G=0

import mediapipe as mp


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


cap=cv2.VideoCapture('pluie.mp4')

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
   
size = (frame_width, frame_height)
   
# Below VideoWriter object will create
# a frame of above defined The output 
# is stored in 'filename.avi' file.
result = cv2.VideoWriter('detection1.avi', 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         30, size)
while True:


    ret,image=cap.read()
    print(model.predict(image, confidence=40, overlap=30).json())
    predictions=model.predict(image, confidence=40, overlap=30).json()
    predictions1=model1.predict(image, confidence=40, overlap=30).json()
    print(model1.predict(image, confidence=40, overlap=30).json())

    detections = predictions['predictions']
    detections1 = predictions1['predictions']

    for box in detections:
        x11 =int( box['x'] - box['width'] / 2)
        x21 =int( box['x'] + box['width'] / 2)
        y11 = int(box['y'] - box['height'] / 2)
        y21 = int (box['y'] + box['height'] / 2)
        cv2.rectangle(image,(x11,y11),(x21,y21),(255,0,0),2)
        cv2.putText(image, 'FIRE', (x21, y21), cv2.FONT_HERSHEY_TRIPLEX, 1, (255,255,255), 2)

    
    A=predictions1['predictions']
    if A!=[]:
     B=A[0]
     C=B['class']
    else:
     C=1

    if (C=='chute'):
        
     for box in detections1:
        
        x111 =int( box['x'] - box['width'] / 2)
        x211 =int( box['x'] + box['width'] / 2)
        y111 = int(box['y'] - box['height'] / 2)
        y211 = int (box['y'] + box['height'] / 2)
        cv2.rectangle(image,(x111,y111),(x211,y211),(255,0,0),5)
        cv2.putText(image, 'FALL', (x211, y211), cv2.FONT_HERSHEY_TRIPLEX, 1, (255,255,255), 2)


    with mp_pose.Pose(
        min_detection_confidence=0.2,
        min_tracking_confidence=0.2) as pose:
    
        
        

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        h,w,z = image.shape
        lm = results.pose_landmarks

        if results.pose_landmarks:
            x1 = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * w)
            y1 = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * h)

            x2 = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x * w)
            y2 = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * h)

            x3 = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].x * w)
            y3 = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y * h)

            x4 = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].x * w)
            y4 = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y * h)
            
            
            if y4 < y2:
                j=j+1
                if j>5:
                 #print("FALL")
                 cv2.putText(image, 'FALL', (x2, y2), cv2.FONT_HERSHEY_TRIPLEX, 1, (255,255,255), 2)
                 cv2.rectangle(image,(x2,y2),(x4,y4),(255,0,0),4)
                 G=1
                else:
                    cv2.rectangle(image,(x2,y2),(x4,y4),(0,255,255),2)

            else:
                G=0
                j = 0 
                cv2.rectangle(image,(x2,y2),(x4,y4),(0,255,255),2)

        # Draw the pose annotation on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        '''mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())'''
        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('video',image)
        result.write(image)
        if cv2.waitKey(4) & 0xFF==ord("q"):
            break
cap.release()
cv2.destroyAllWindows()

