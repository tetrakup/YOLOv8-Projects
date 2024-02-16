#from email.policy import default
import cv2
import imutils #elimizdeki fotoğrafı yeniden boyutlandırmak için: en ve boy oranı

import numpy as np
from ultralytics import YOLO
from collections import defaultdict

color = (0,255,0)
color_red = (0,0,255)
thickness = 2

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.5

video_path = "inference/test.mp4"
model_path = "models/yolov8n.pt"

cap = cv2.VideoCapture(video_path) #videoyu okumak için
model = YOLO(model_path) #modelimizi dahil etme

#kayit islemleri icin
width = 1280
height = 720

fourcc = cv2.VideoWriter_fourcc(*'XVID')
writer = cv2.VideoWriter("video.avi", fourcc, 20.0, (width,height))

vehicle_ids = [2, 3, 5, 7] #coco-classes.txt'den alınan takip etmek istediğmiz nesne id'leri
track_history = defaultdict(lambda: []) #araclarin gidis y onu tespiti icin tails

up = {}
down = {}
threshold = 450

while True: #görüntüyü okumayı deneyeceğiz.
    ret, frame = cap.read() #videoyu okuduk.
    if ret == False:
        break
      
    frame = imutils.resize(frame, width = 1280)#işlemek için kull. frame  
    results = model.track(frame, persist=True, verbose=False)[0] #model.track denildiğinde yolov8'in takip modülü calisiyor. / Verbose her çıkıtıyı term. yazdirma islemi. /persist: frame'lar arası nesne takibi

    #track_ids = results.boxes.id.int().cpu().tolist() #id.it: id int foramtında iletir. /cpu.tolist: cpu'yu list formatında 
    bboxes = np.array(results.boxes.data.tolist(), dtype="int") #xyxy

    cv2.line(frame, (0,threshold), (1280,threshold), color, thickness) #bu referans çizgisi gecildiyse arac sayimi yapilacak
    cv2.putText(frame, "Reference Line", (620, 445), font, 0.7, color_red, thickness)
    
    for box in bboxes:
        x1, y1, x2, y2, track_id, score, class_id = box #x1, y1=dikdörtgenin sol üst köşesi,x2, y2:sag alt kösesi
        cx = int((x1+x2)/2) #merkez hesaplama
        cy = int((y1+y2)/2)
        if class_id in vehicle_ids:
            class_name = results.names[int(class_id)].upper()  #class_name'lere eriştik. float olarak dönmemesi için int. çevirdik.
        # print("BBoxes: ",(x1, y1, x2, y2))
        # print("Class: ", class_name)
        # print("ID: ", track_id)
            

        track = track_history[track_id]
        track.append((cx, cy)) #kordinatlari depoluyorz
        if len(track) > 20: #eger kuyruk sayisi 20den fazlaysa sifirla.
            track.pop(0)

        points = np.hstack(track).astype("int32").reshape(-1,1,2) #yatay olarak yanyana sıralamak icin, eshape(-1,1,2) 3b diziye dönüstürür
        cv2.polylines(frame, [points], isClosed=False, color=color, thickness=thickness)
        cv2.rectangle(frame, (x1,y1), (x2,y2), color, thickness)

        text = "ID: {} {}".format(track_id, class_name)
        cv2.putText(frame, text, (x1, y1-5), font, font_scale, color, thickness)
        
        if cy>threshold-5 and cy<threshold +5 and cx<670:
            down[track_id] = x1,y1, x2, y2
        
        if cy>threshold-5 and cy<threshold +5 and cx>670:
            up[track_id] = x1,y1, x2, y2
        
    print("UP Dictionary Keys:", list(up.keys()))
    print("DOWN Dictionary Keys:", list(down.keys()))

    up_text = "Giden:{}".format(len(list(up.keys())))
    down_text = "Gelen:{}".format(len(list(down.keys())))

    cv2.putText(frame, up_text, (1150, threshold-5), font, 0.8, color_red, thickness)
    cv2.putText(frame, down_text, (0, threshold-5), font, 0.8, color_red, thickness)

    writer.write(frame)
    #görüntüyü gösterdiğimiz yer
    cv2.imshow("Test", frame) # ilk parametre penc. ismi
    if cv2.waitKey(10) & 0xFF==ord("q"): #q'ya basılınca break olacak.
        break

cap.release() #çıktıktan sonra video serbest birakilmali.
writer.release()
cv2.destroyAllWindows()

print("[INFO]..The video was succesfully precessed/saved!")


