import cv2 as cv
import time
import numpy as np

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

'''confThreshold = 0.5  #Confidence threshold
disThreshold = 200
inpWidth = 320       #Width of network's input image
inpHeight = 320      #Height of network's input image'''

def load_net(cfg_path, net_weights): #configuration file path, network weights file path
    #loads network
    net = cv.dnn.readNetFromDarknet(cfg_path, net_weights)
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_OPENCL)
    
    return net #returns network object

#performs detection with neural network
def detect(net, im, confThreshold = 0.5, inpWidth=320, inpHeight=320):
    in_blob = cv.dnn.blobFromImage(im, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)

    net.setInput(in_blob)

    pred = net.forward(getOutputsNames(net))
    pred = np.array(pred) #converts into numpy array
    pred = np.concatenate([np.array(i) for i in pred]) #reshapes into 1 dimension array

    final_im = postprocess(im, pred)
    return final_im #returns post processed result - center_x and center_y

#gets name of layers with unconnected outputs, i.e. output layers
def getOutputsNames(net):
    layersNames = net.getLayerNames()
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

#postprocesses network results -- nms, results post predictions
def postprocess(frame, pred, confThreshold=0.5):
    fHeight = frame.shape[0] #height of frame
    fWidth = frame.shape[1] #width of frame
    
    pred = np.array([x for x in pred if x[5] >= confThreshold]) #removes confidences under threshold
    
    post_pred = [] #list of all post predictions
    
    logger.debug("Frame Height: {}".format(fHeight))
    logger.debug("Frame Width: {}".format(fWidth))
    
    while pred.size > 0:
        hc_i = np.argmax(pred, axis=0)[5] #highest confidence index
        hc_row = pred[hc_i] #highest confidence row

        center_x = int(hc_row[0] * fWidth)
        center_y = int(hc_row[1] * fHeight)
        width = int(hc_row[2] * fWidth)
        post_pred.append((center_x, center_y)) #first element to base nms off of
        
        pred = np.array([x for x in pred if checkOverlap(center_x, width, int(x[0] * fWidth))])
    
    #print(post_pred)
    return post_pred

#used in postprocess nms -- the nms threshold is 0, any overlapping boxes can be deleted
def checkOverlap(x_box, w_box, x_test):
    return not (x_box - w_box / 2) <= x_test <= (x_box + w_box / 2) #if x coordinate is overlapping deletes that stuff

def pred_circle(x_coord, y_coord, im, color): #color is rgb tuple
    cv.circle(im, (x_coord, y_coord), 10, color, 4)
    
    #shows image
    cv.imshow('frame', im)
    cv.waitKey(0)
    cv.destroyAllWindows()

def main():
    net = load_net("yolov3-tiny-obj.cfg", "yolov3-tiny-obj_final.weights")
    test_im = cv.imread("testImage.png")
    b_info = detect(net, test_im)
    
    logger.info("ball information: {}".format(b_info))
    
    pred_circle(b_info[0][0], b_info[0][1], test_im, (255, 0, 0))

if __name__ == "__main__":
    main()