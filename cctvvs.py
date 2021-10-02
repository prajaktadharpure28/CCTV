import numpy as np
import cv2
import datetime
from datetime import datetime
import time
from calendar import timegm

d = datetime.utcnow()
d = d.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
utc_time = time.strptime(d,"%Y-%m-%dT%H:%M:%S.%fZ")

filename = 'output_{0}.avi'.format(utc_time)


hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cv2.startWindowThread()


cap = cv2.VideoCapture(0)


out = cv2.VideoWriter(
    filename,
    cv2.VideoWriter_fourcc(*'MJPG'),
    5,
    (640, 480))

while (True):
    
    ret, frame = cap.read()

    
    frame = cv2.resize(frame, (640, 480))
    
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    
    boxes, weights = hog.detectMultiScale(frame, winStride=(8, 8))

    boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])

    for (xA, yA, xB, yB) in boxes:
        
        cv2.rectangle(frame, (xA, yA), (xB, yB),
                      (0, 255, 0), 2)
        
        out.write(frame.astype('uint8'))
    
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()

out.release()

cv2.destroyAllWindows()
cv2.waitKey(1)
