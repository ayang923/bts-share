import cv2 as cv
import time
import numpy as np
import math

import logging

from functions import load_net, detect, pred_circle

import matplotlib.pyplot as plt


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

video_source = "testVid.mov"

confThreshold = 0.3
disThreshold = 200

def vel_pred(detections, prev_X, prev_T):
    Xtime = time.time() - startTime
    if len(detections) != 0:
        for detection in detections:
            delta_t = Xtime - prev_X
            logger.info("Change in time: {}".format(delta_t))
            P = prev_X+v*delta_t
                    
            #update v and return it
            v = (detection[0] - prev_X)/(delta_t)

    return v


def main():
    net = load_net("yolov3-tiny-obj.cfg", "yolov3-tiny-obj_final.weights")
    
    cap = cv.VideoCapture(video_source)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    count = 0 #number of iterations
    X = [] #xpositions
    T = [] #times
    P = [] #predictions

    startTime = time.time()
    v = 0 #velocity
    while True:
        #reads frame
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        #performs detection
        detections = detect(net, frame, confThreshold)
        Xtime = time.time() - startTime
        if len(detections) != 0:
            for detection in detections:
                if len(X) >= 2:
                    delta_t = Xtime - T[-1]
                    logger.info(delta_t)
                    P.append(X[-1]+v*delta_t)
                    
                    #update v
                    v = (detection[0] - X[-1])/(delta_t)

                #saving X vs T in present time
                X.append(detection[0])
                T.append(Xtime)
                
                #velocity calculation and predicitons
                

        #Display the resulting frame
        for detection in detections:
            pred_circle(detection[0], detection[1], frame, (255, 0, 0))
        logger.info("Detection: {}".format(detections))
        cv.imshow('frame', frame)
        if cv.waitKey(1) == ord('q'):
            break

        count += 1 #increase count

    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()
    
    logger.debug("X: {}".format(X))
    logger.info("T: {}".format(len(T[2:])))
    logger.info("P: {}".format(len(P)))
    
    plt.figure(figsize=(12, 5))
    plt.scatter(T, X)
    plt.scatter(T[2:], P, c='r')
    plt.xlabel("Time")
    plt.ylabel("X Position")
    plt.title("Time vs X Position")
    plt.show()
    
if __name__ == '__main__':
    main()