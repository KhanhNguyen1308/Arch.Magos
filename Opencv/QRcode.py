import cv2
import numpy as np
from pyzbar.pyzbar import decode
cap = cv2.VideoCapture(0)
while True:
    fet, frame = cap.read()
    frame = cv2.resize(frame, dsize=None, fx=0.5, fy=0.5)
    code = decode(frame)
    for barcode in code:
        mydata = barcode.data.decode('utf-8')
        print(mydata)
        pts = np.array([barcode.polygon], np.int32)
        pts = pts.reshape((-1,1,2))
        cv2.polylines(frame, [pts], True, (255, 0, 255), 2)
        x, y, w, h = barcode.rect
        # cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 255), 2)
        bdata = barcode.data.decode('utf-8')
        print(mydata)
        btype = barcode.type
        text = f"{bdata},{btype}"
        cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imshow("frame", frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    if key == ord('a'):
        print(code)

cap.release()
cv2.destroyAllWindows()