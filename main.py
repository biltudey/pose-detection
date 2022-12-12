import cv2
import mediapipe as mp
import numpy as np

cap = cv2.VideoCapture('6.mp4')

mpose = mp.solutions.pose
pose =mpose.Pose()

draw = mp.solutions.drawing_utils
draw1=draw.DrawingSpec(thickness=2,circle_radius=3,color=(0,0,255))
draw2=draw.DrawingSpec(thickness=2,circle_radius=3,color=(0,255,0))

while True:
    succes , img = cap.read()
    img = cv2.resize(img,(600,600))

    result = pose.process(img)
    draw.draw_landmarks(img,result.pose_landmarks,mpose.POSE_CONNECTIONS,draw1,draw2)

    h,w,c = img.shape
    imgBlank = np.zeros([h,w,c])
    imgBlank.fill(255)
    draw.draw_landmarks(imgBlank,result.pose_landmarks,mpose.POSE_CONNECTIONS,draw1,draw2)


    cv2.imshow('org',img)
    cv2.imshow('Pose',imgBlank)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break