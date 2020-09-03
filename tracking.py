import numpy as np
import cv2

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') # choose the classifier

cap = cv2.VideoCapture(0)
ret = cap.set(3,1000) # (3,x) - width / (4,y) - height, so 640x480 by default ~ it's size modification of resulting window
while True: #while frame is read correctly it'll be True value
    ret, frame = cap.read(0) # capture frames one by one ~ argument is webcam index on your machine
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # convert to grayscale as many functions expect grayscale in opencv and other libs

    # scaleFactor is the pace for resizing by comparison with the base model (the lower the value, the more thorough and slow the approach)
    # minNeighbors is the parameter which determines how many faces can be detected (the higher value results in fewer detections, but with higher quality)
    # minSize is the minimum possible size of the faces (objects smaller than they are ignored)
    track = faceCascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=6, minSize=(30, 30))

    for (x, y, w, h) in track: # from basic opencv multiscale detection we get faces centers coordinates and also the widths and the heights of the faces
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2) # to write rectangle in opencv we need top-left and bottom-right corners ~ i'll use yellow clr

    cv2.imshow('TRACKING in progress - type <q> or <Q> for exit', frame) # display the resulting frames in window with name "TRACKING"

    if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')): # wait for q/Q (quit) key press to exit ~ i need modify this option by 0xFF as i'm using a 64-bit machine
        break

cap.release()
cv2.destroyAllWindows() #destroy all windows which we've already created