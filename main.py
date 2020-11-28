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
angleThreshold = 30 #difference in angles
exmpt_disThreshold = 50 #exempts noise

def vel_pred(detection, prev_x, prev_t, t, v):
    delta_t = t - prev_t
    logger.info("Change in time: {}".format(delta_t))

    #update v and p and return it
    v = (detection[0] - prev_x)/(delta_t)
    p = detection[0]+v*delta_t

    logger.info("v:{}\np:{}\nt:{}".format(v, p, t))

    return v, p #returns velocity and prediction

def angle(x, y, z): #tuples with coordinates
    yx = np.subtract(x, y)
    yz = np.subtract(z, y)
    
    cosa = np.dot(yx, yz)/(np.linalg.norm(yx)*np.linalg.norm(yz))
    a = math.degrees(math.acos(cosa))
    
    return a

def coord_norm(x, y, z, frame_x):
    xyz = np.vstack((x, y, z))[:, 1]
    
    low = xyz.min()
    high = xyz.max()
    r = high - low #range

    normalized = ((0, (x[1])/frame_x), ((y[0]-x[0])/(z[0]-x[0]), (y[1])/frame_x), (1, (z[1])/frame_x))
    
    return normalized

def angle_prediction(coord1, coord2, coord3, pred, fwidth):
    x_norm, y_norm, z_norm = coord_norm(coord1, coord2, coord3, fwidth)
    xp_norm, yp_norm, zp_norm = coord_norm(coord1, pred, coord3, fwidth)

    a_actual = angle(x_norm, y_norm, z_norm)
    a_pred = angle(xp_norm, yp_norm, zp_norm)
    delta_angle = a_actual - a_pred
    logger.info("actual ({}) - prediction ({}) = {}".format(a_actual, a_pred, delta_angle))
    
    return delta_angle

def main():
    net = load_net("yolov3-tiny-obj.cfg", "yolov3-tiny-obj_final.weights")
    
    cap = cv.VideoCapture(video_source)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    count = 0 #number of iterations
    X = [] #xpositions
    P = [] #predictions
    T = [] #times
    delta_angles = [] #angle differences

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
        
        logger.info("\nCount: {}\n".format(len(X)))
        
        if len(detections) != 0:
            T.append(time.time() - startTime) #appends time

            index = 0
            if len(X) % 2 == 0 and len(X) >= 4:
                angles = [angle((T[-3], X[-2]), (T[-2], P[-1]), (T[-1], detection[0])) for detection in detections]
                
                index = angles.index(max(angles))
                
                logger.info("Angles: {}".format(angles))
                
            if len(X) % 2 == 1 and len(X) >= 3:
                distances = [P[-1] - detection[0] for detection in detections]
                
                index = distances.index(min(distances))
                logger.info("Distances: {}".format(distances))
                
            X.append(detections[index][0])
            logger.info("Detection: {}".format(detections))
 
        if len(X) >= 2 and len(detections) != 0:
        
            
            if len(X) % 2 == 0:
                if len(X) >=4:   
                    delta_angle = angle_prediction((T[-3], X[-3]), (T[-2], X[-2]), (T[-1], X[-2]), (T[-2], P[-1]), 1280)
                    if delta_angle < -angleThreshold:
                        X[-2] = P[-1]

                v, p = vel_pred(detections[0], X[-2], T[-2], T[-1], v)
                P.append(p) # velocity predictions
                
            
        #Display the resulting frame
        for detection in detections:
            pred_circle(detection[0], detection[1], frame, (255, 0, 0))
        cv.imshow('frame', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

        count += 1 #increase count

    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()
    
    logger.info("X: {}".format(len(X)))
    logger.info("T: {}".format(len(T)))
    logger.info("P: {}".format(len(P)))
    logger.info("Angles: {}".format(len(delta_angles)))
    
    plt.figure(figsize=(12, 5))
    plt.scatter(T, X)
    #plt.scatter(T[2::2], P[:-1], c='r')
    #plt.scatter(T[2::2], delta_angles, c='g')
    plt.xlabel("Time")
    plt.ylabel("X Position")
    plt.title("Time vs X Position")
    plt.show()
    
if __name__ == '__main__':
    main()