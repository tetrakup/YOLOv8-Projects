import os 
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import imutils

import numpy as np
from ultralytics import YOLO


blue = (255,0,0)
green = (0,255,0)
red = (0,0,255)
black = (0,0,0)
white = (255,255,255)

thickness = 2
font_scale = 0.6
font = cv2.FONT_HERSHEY_SIMPLEX

polygon_1_dict = {}
polygon_2_dict = {}
polygon_3_dict = {}
polygon_4_dict = {}
polygon_5_dict = {}
polygon_6_dict = {}
#artı dokuz eklenecek
polygon_7_dict = {}
polygon_8_dict = {}
polygon_9_dict = {}
polygon_10_dict = {}
polygon_11_dict = {}
polygon_12_dict = {}
polygon_13_dict = {}
polygon_14_dict = {}
polygon_15_dict = {}

vehicles = [2, 3, 5, 7]

polygon_1 = np.array([[634, 471],[629, 525],
                      [767, 523],[739, 465]], np.int32)

polygon_2 = np.array([[752, 464],[785, 521],
                      [879, 516],[820, 451]], np.int32)

polygon_3 = np.array([[897, 516],[837, 450],
                      [886, 441],[968, 507]], np.int32) 

polygon_4 = np.array([[981, 505],[898, 437],
                      [954, 430],[1049, 497]], np.int32)

polygon_5 = np.array([[1060, 493],[965, 425],
                      [1018, 421],[1118, 483]], np.int32)

polygon_6 = np.array([[1127, 481],[1030, 417],
                       [1093, 408],[1181, 467]], np.int32)
#7-15
import numpy as np

polygon_7 = np.array([[354, 521], [433, 525], [413, 473], [462, 482]], np.int32)

polygon_8 = np.array([[449, 523], [529, 527], [484, 480], [548, 479]], np.int32)

polygon_9 = np.array([[193, 416], [213, 419], [241, 387], [266, 392]], np.int32)

polygon_10 = np.array([[253, 418], [312, 419], [305, 400], [348, 399]], np.int32)

polygon_11 = np.array([[319, 421], [350, 423], [355, 401], [384, 401]], np.int32)

polygon_12 = np.array([[363, 423], [401, 423], [398, 401], [430, 401]], np.int32)

polygon_13 = np.array([[411, 423], [449, 427], [439, 402], [473, 402]], np.int32)

polygon_14 = np.array([[458, 428], [502, 428], [489, 402], [529, 401]], np.int32)

polygon_15 = np.array([[517, 426], [557, 431], [541, 394], [579, 394]], np.int32)


video_path = "inference/test.mp4"
cap = cv2.VideoCapture(video_path)
model = YOLO("models/yolov8n.pt")


width = 1280
height = 720

fourcc = cv2.VideoWriter_fourcc(*'XVID')
writer = cv2.VideoWriter("video.avi", fourcc, 20.0, (width, height))

free_space_counter = []


while True:
    ret, frame = cap.read()
    if ret==False:
        break
    
    frame = imutils.resize(frame, width=1280)
    frame_copy = frame.copy()
    
    cv2.polylines(frame_copy, [polygon_1], isClosed=True, color=green, thickness=thickness)
    cv2.polylines(frame_copy, [polygon_2], isClosed=True, color=green, thickness=thickness)
    cv2.polylines(frame_copy, [polygon_3], isClosed=True, color=green, thickness=thickness)
    cv2.polylines(frame_copy, [polygon_4], isClosed=True, color=green, thickness=thickness)
    cv2.polylines(frame_copy, [polygon_5], isClosed=True, color=green, thickness=thickness)
    cv2.polylines(frame_copy, [polygon_6], isClosed=True, color=green, thickness=thickness)
    #7-15
    cv2.polylines(frame_copy, [polygon_7], isClosed=True, color=green, thickness=thickness)
    cv2.polylines(frame_copy, [polygon_8], isClosed=True, color=green, thickness=thickness)
    cv2.polylines(frame_copy, [polygon_9], isClosed=True, color=green, thickness=thickness)
    cv2.polylines(frame_copy, [polygon_10], isClosed=True, color=green, thickness=thickness)
    cv2.polylines(frame_copy, [polygon_11], isClosed=True, color=green, thickness=thickness)
    cv2.polylines(frame_copy, [polygon_12], isClosed=True, color=green, thickness=thickness)
    cv2.polylines(frame_copy, [polygon_13], isClosed=True, color=green, thickness=thickness)
    cv2.polylines(frame_copy, [polygon_14], isClosed=True, color=green, thickness=thickness)
    cv2.polylines(frame_copy, [polygon_15], isClosed=True, color=green, thickness=thickness)
    
    

    results = model.track(frame, persist=True, verbose=False)[0]
    bboxes = np.array(results.boxes.data.tolist(), dtype="int")
    
    for box in bboxes:
        x1, y1, x2, y2, track_id, score, class_id = box
        cx = int((x1+x2)/2)
        cy = int((y1+y2)/2) #aracların merkez koordinatları
        
        if class_id in  vehicles:
            cv2.circle(frame_copy, (cx, cy), 3, blue, -1)
            cv2.rectangle(frame_copy, (x1,y1), (x2,y2), blue, thickness=1)          
            
            polygon_1_result = cv2.pointPolygonTest(polygon_1, (cx, cy), measureDist=False) #0dan büyük,1 dönerse arac geçmiştir diyebiliriz
            polygon_2_result = cv2.pointPolygonTest(polygon_2, (cx, cy), measureDist=False)
            polygon_3_result = cv2.pointPolygonTest(polygon_3, (cx, cy), measureDist=False)
            polygon_4_result = cv2.pointPolygonTest(polygon_4, (cx, cy), measureDist=False)
            polygon_5_result = cv2.pointPolygonTest(polygon_5, (cx, cy), measureDist=False)
            polygon_6_result = cv2.pointPolygonTest(polygon_6, (cx, cy), measureDist=False)
            #7 -15
            polygon_7_result = cv2.pointPolygonTest(polygon_7, (cx, cy), measureDist=False)
            polygon_8_result = cv2.pointPolygonTest(polygon_8, (cx, cy), measureDist=False)
            polygon_9_result = cv2.pointPolygonTest(polygon_9, (cx, cy), measureDist=False)
            polygon_10_result = cv2.pointPolygonTest(polygon_10, (cx, cy), measureDist=False)
            polygon_11_result = cv2.pointPolygonTest(polygon_11, (cx, cy), measureDist=False)
            polygon_12_result = cv2.pointPolygonTest(polygon_12, (cx, cy), measureDist=False)
            polygon_13_result = cv2.pointPolygonTest(polygon_13, (cx, cy), measureDist=False)
            polygon_14_result = cv2.pointPolygonTest(polygon_14, (cx, cy), measureDist=False)
            polygon_15_result = cv2.pointPolygonTest(polygon_15, (cx, cy), measureDist=False)

            
            if polygon_1_result >= 0:
                 #print("Polygon 1!")
                 polygon_1_dict[track_id] = x1, y1, x2, y2
                 free_space_counter.append(polygon_1_dict)
                # cv2.putText(frame_copy, "Not Free", (cx, cy), font, 0.8, red, thickness)

            if polygon_2_result >= 0:
                 #print("Polygon 2!")
                 polygon_2_dict[track_id] = x1, y1, x2, y2
                 free_space_counter.append(polygon_2_dict)
                # cv2.putText(frame_copy, "Not Free", (cx, cy), font, 0.8, red, thickness)
                 
            if polygon_3_result >= 0:
                 #print("Polygon 3!")
                 polygon_3_dict[track_id] = x1, y1, x2, y2
                 free_space_counter.append(polygon_3_dict)
                # cv2.putText(frame_copy, "Not Free", (cx, cy), font, 0.8, red, thickness)
                 
            if polygon_4_result >= 0:
                 #print("Polygon 4!")
                 polygon_4_dict[track_id] = x1, y1, x2, y2
                 free_space_counter.append(polygon_4_dict)
                # cv2.putText(frame_copy, "Not Free", (cx, cy), font, 0.8, red, thickness)
                 
            if polygon_5_result >= 0:
                 #print("Polygon 5!")
                 polygon_5_dict[track_id] = x1, y1, x2, y2
                 free_space_counter.append(polygon_5_dict)
                # cv2.putText(frame_copy, "Not Free", (cx, cy), font, 0.8, red, thickness)
                 
            if polygon_6_result >= 0:
                 #print("Polygon 6!")
                 polygon_6_dict[track_id] = x1, y1, x2, y2
                 free_space_counter.append(polygon_6_dict)
                # cv2.putText(frame_copy, "Not Free", (cx, cy), font, 0.8, red, thickness)
                 
                 #7-15
            if polygon_7_result >= 0:
                 #print("Polygon 1!")
                 polygon_7_dict[track_id] = x1, y1, x2, y2
                 free_space_counter.append(polygon_7_dict)
                # cv2.putText(frame_copy, "Not Free", (cx, cy), font, 0.8, red, thickness)
                 
            if polygon_8_result >= 0:
                 #print("Polygon 1!")
                 polygon_8_dict[track_id] = x1, y1, x2, y2
                 free_space_counter.append(polygon_8_dict)
                # cv2.putText(frame_copy, "Not Free", (cx, cy), font, 0.8, red, thickness)
                 
            if polygon_9_result >= 0:
                 #print("Polygon 1!")
                 polygon_9_dict[track_id] = x1, y1, x2, y2
                 free_space_counter.append(polygon_9_dict)
                # cv2.putText(frame_copy, "Not Free", (cx, cy), font, 0.8, red, thickness)
            
            if polygon_10_result >= 0:
                 #print("Polygon 1!")
                 polygon_10_dict[track_id] = x1, y1, x2, y2
                 free_space_counter.append(polygon_10_dict)
                # cv2.putText(frame_copy, "Not Free", (cx, cy), font, 0.8, red, thickness)
            
            if polygon_11_result >= 0:
                 #print("Polygon 1!")
                 polygon_11_dict[track_id] = x1, y1, x2, y2
                 free_space_counter.append(polygon_11_dict)
                # cv2.putText(frame_copy, "Not Free", (cx, cy), font, 0.8, red, thickness)
            
            if polygon_12_result >= 0:
                 #print("Polygon 1!")
                 polygon_12_dict[track_id] = x1, y1, x2, y2
                 free_space_counter.append(polygon_12_dict)
                # cv2.putText(frame_copy, "Not Free", (cx, cy), font, 0.8, red, thickness)
            
            if polygon_13_result >= 0:
                 #print("Polygon 1!")
                 polygon_13_dict[track_id] = x1, y1, x2, y2
                 free_space_counter.append(polygon_13_dict)
                # cv2.putText(frame_copy, "Not Free", (cx, cy), font, 0.8, red, thickness)
            
            if polygon_14_result >= 0:
                 #print("Polygon 1!")
                 polygon_14_dict[track_id] = x1, y1, x2, y2
                 free_space_counter.append(polygon_14_dict)
                # cv2.putText(frame_copy, "Not Free", (cx, cy), font, 0.8, red, thickness)
            
            if polygon_15_result >= 0:
                 #print("Polygon 1!")
                 polygon_15_dict[track_id] = x1, y1, x2, y2
                 free_space_counter.append(polygon_15_dict)
                # cv2.putText(frame_copy, "Not Free", (cx, cy), font, 0.8, red, thickness)
    
            
    free_lot = (6-len(free_space_counter))
    free_space_counter.clear()

    text = "Free Parkign Lot: {}".format(free_lot)
    cv2.putText(frame_copy, text, (10, 25), font, 0.8, black, thickness)
    
    writer.write(frame_copy)
    cv2.imshow("Free Parking Space Counter", frame_copy)
    if cv2.waitKey(10) & 0xFF==ord("q"):
        break
    
cap.release()
writer.release()

cv2.destroyAllWindows()
print("[INFO]...Processing is finished!")
