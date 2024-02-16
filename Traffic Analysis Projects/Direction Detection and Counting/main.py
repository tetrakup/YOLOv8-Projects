import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import imutils

import numpy as np
from ultralytics import YOLO
from collections import defaultdict

blue = (255,0,0)
green = (0,255,0)
red = (0,0,255)
black= (0,0,0)

thickness = 2
font_scale = 0.6
font = cv2.FONT_HERSHEY_SIMPLEX
up = {}
down = {}
left = {}
right = {}

vehicles = [2,3,5,7]

polygon_up = np.array([[467, 295],[578, 370],
                       [710, 361],[559, 283]], np.int32)

polygon_down = np.array([[760, 610],[490, 660],
                         [527, 714],[890, 711]], np.int32)

polygon_left = np.array([[107, 451],[213, 438],
                         [220, 469],[85, 481]], np.int32) # ÖDEV: Alanı düzenle.

polygon_right = np.array([[990, 453],[1262, 404],
                          [1276, 467],[1072, 506]], np.int32)


video_path = "inference/intersection.mp4"
cap = cv2.VideoCapture(video_path)

while True:
    ret, frame = cap.read()
    if ret==False:
        break
    
    frame = imutils.resize(frame, width=1280)

    cv2.polylines(frame, [polygon_up], isClosed=True,color=green, thickness=thickness)
    cv2.polylines(frame, [polygon_down], isClosed=True,color=green, thickness=thickness)
    cv2.polylines(frame, [polygon_left], isClosed=True,color=green, thickness=thickness)
    cv2.polylines(frame, [polygon_right], isClosed=True,color=green, thickness=thickness)
    

    cv2.imshow("Direction Counter", frame)
    if cv2.waitKey(10) & 0xFF==ord("q"): 
        break

cap.release()

cv2.destroyAllWindows()

