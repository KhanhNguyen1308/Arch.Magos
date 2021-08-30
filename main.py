import cv2
import time
from function import count_object


CONFIDENCE_THRESHOLD = 0.2
NMS_THRESHOLD = 0.4
COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]

class_names = []
with open("classes.txt", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]

vc = cv2.VideoCapture("aespa.mp4")

net = cv2.dnn.readNetFromDarknet("cfg/yolov4-tiny-256-2.cfg", "data/yolov4-tiny-256-2.weights")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(256, 256), scale=1/255, swapRB=True)
resize = False
while cv2.waitKey(1) < 1:
    (grabbed, frame) = vc.read()
    if not grabbed:
        exit()

    start = time.time()
    classes, scores, boxes = model.detect(frame, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    face, person = count_object(classes)
    end = time.time()

    start_drawing = time.time()
    for (classid, score, box) in zip(classes, scores, boxes):
        color = COLORS[int(classid) % len(COLORS)]
        label = "%s : %f" % (class_names[classid[0]], score)
        cv2.rectangle(frame, box, color, 2)
        cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    end_drawing = time.time()
    
    fps_label = "FPS: %.2f (excluding drawing time of %.2fms)" % (1 / (end - start), (end_drawing - start_drawing) * 1000)
    cv2.putText(frame, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(frame, "face: "+str(face), (10,40), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
    cv2.putText(frame, "person: "+str(person), (10,60), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
    if resize:
        frame = cv2.resize(frame, dsize=None, fx=0.5, fy=0.5)
    cv2.imshow("detections", frame)
    key = cv2.waitKey(1)
    if key==ord('q'):
        break
    if key == ord('a'):
        resize = True

vc.release()
cv2.destroyAllWindows()