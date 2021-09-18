import cv2
import time

i=3
CONFIDENCE_THRESHOLD = 0.2
NMS_THRESHOLD = 0.4
COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0), (255, 0, 255),(125, 0, 125), (125, 125, 0), (0, 125, 125)]
class_names = []
with open("classes.txt", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]
vc = cv2.VideoCapture("A.mp4")
#vc = cv2.VideoCapture(0)
net = cv2.dnn.readNetFromDarknet("cfg/6-class.cfg", "model/6-class.weights")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)
resize = False
while True:
    (grabbed, frame) = vc.read()
    original = frame.copy()
    if frame is None:
        print("None frame to cap!")
        print("Exit!")
        break

    start = time.time()
    classes, scores, boxes = model.detect(frame, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    end = time.time()
    start_drawing = time.time()
    for (classid, score, box) in zip(classes, scores, boxes):
        color = COLORS[int(classid) % len(COLORS)]
        label = "%s : %f" % (class_names[classid[0]], score)
        cv2.rectangle(frame, box, color, 2)
        cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    end_drawing = time.time()
    fps_label = "FPS: " + str(int(1 / (end - start)))
    cv2.putText(frame, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    if resize:
        frame = cv2.resize(frame, dsize=None, fx=0.5, fy=0.5)
    cv2.imshow("detections", frame)
    key = cv2.waitKey(1)
    if key==ord('q'):
        break
    if key == ord('a'):
        resize = True
    if key == ord('x'):
        cv2.imwrite("new_img/z"+str(i)+".jpg",original)
        i+=1

vc.release()
cv2.destroyAllWindows()
