import numpy as np
import tensorflow as tf
import cv2

mod_path = "yolov3-tiny.tflite"
img_path = "testImage.png"

def process_image(image_path, input_height, input_width):
    """
    Takes any image and transforms it into the format needed for object detection with yolov3.
    Parameters
    ----------
    image_path : string
        Path that points to where the image on which object detection should be performed is stored.
    input_height : int
        The height of the input that will be fed into the yolov3 model.
    input_width : int
        The width of the input that will be fed into the yolov3 model.
    
    Returns
    -------
    resized_image : ndarray
        An array of shape:
        [input_height, input_wdith, 3]
        The array is divided by 255.0 in order to turn the pixel values into numbers between zero 
        and one. Since cv2 load images in BGR, the array is also converted to a RGB color profile.
    image : ndarray
        An array of shape:
        [image height, image width, 3]
        The original image simply loaded into a numpy array with a RGB color profile.
    """

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized_image = cv2.resize(image,(input_width,input_height))
    resized_image = resized_image / 255.0

    return resized_image, image

def convert_box_coordinates(detections):
    """
    Converts coordinates in the form of center_x, center_y, width, height to 
    min_x, min_y, max_x, max_y. The coordinate values are already scaled up to 
    the input dimension shapes.
    Parameters
    ----------
    detections : ndarray
        An array of shape:
        [1, num_large_obj_detectors + num_med_obj_detectors + num_small_obj_detectors, num_classes + 5]
        where num_x_obj_detectors = num_anchors_per_layer * yolo_layer_grid_w * yolo_layer_grid_h. 
    Returns
    -------
    detections : ndarray
        The original detections array with converted coordinates.
    """

    split = np.array_split(detections, [1, 2, 3, 4, 85], axis=2)
    center_x = split[0]
    center_y = split[1]
    width = split[2]
    height = split[3]
    attrs = split[4]
    
    w2 = width / 2
    h2 = height / 2
    x0 = center_x - w2
    y0 = center_y - h2
    x1 = center_x + w2
    y1 = center_y + h2

    boxes = np.concatenate([x0, y0, x1, y1], axis=-1)
    detections = np.concatenate([boxes, attrs], axis=-1)
    
    return detections

def non_max_suppression(predictions_with_boxes, confidence_threshold, iou_threshold=0.4):
    """
    Applies non-max suppression to predicted boxes.
    Parameters
    ----------
    predictions_with_boxes : ndarray
        An array of shape:
        [1, num_large_obj_detectors + num_med_obj_detectors + num_small_obj_detectors, num_classes + 5]
        where num_x_obj_detectors = num_anchors_per_layer * yolo_layer_grid_w * yolo_layer_grid_h.
    confidence_threshold : float
        A number between zero and one which indicates the minimum object confidence prediction necessary
        for a particular box's data to not be thrown out. For example, the confidence threshold might be
        set to 0.7 and a detector might predict a box with confidence of 0.8. This detector's box data will
        therefore be put in the 'result' dictionary since it is above the confidence threshold.
    iou_threshold : float
        The threshold for deciding if two boxes overlap.
    Returns
    -------
    result : dictionary
        A dictionary of structure: 
        {unique_class_index : [(box_1_data, box_1_prob),(box_2_data, box_2_prob)], etc...}
        where unique_class_index is the index that corresponds with the class's name, 
        box_x_data is a ndarray of size [4] that contains the box information associated 
        with the class index, and box_x_prob is a float that gives the probability of the box
        being in fact the identified class.
    """

    def iou(box1, box2):
        """
        Calculates the intersection over union (IOU) of two bounding boxes, which is the 
        ratio of the area where the two boxes overlap to the joint area of the boxes as a whole.
        Two perfectly overlapping boxes will have an IOU of 1, while two boxes that don't 
        overlap one another at all will have an IOU of 0.
        Parameters
        ----------
        box1 : ndarray
            Array of shape [x_min, y_min, x_max, y_max].
        box2 : ndarray
            Array of shape [x_min, y_min, x_max, y_max].
      
        Returns
        -------
        iou : float
            The IOU result of the two boxes.
        """

        b1_x0, b1_y0, b1_x1, b1_y1 = box1
        b2_x0, b2_y0, b2_x1, b2_y1 = box2

        int_x0 = max(b1_x0, b2_x0)
        int_y0 = max(b1_y0, b2_y0)
        int_x1 = min(b1_x1, b2_x1)
        int_y1 = min(b1_y1, b2_y1)

        int_area = (int_x1 - int_x0) * (int_y1 - int_y0)

        b1_area = (b1_x1 - b1_x0) * (b1_y1 - b1_y0)
        b2_area = (b2_x1 - b2_x0) * (b2_y1 - b2_y0)

        iou = int_area / (b1_area + b2_area - int_area + 1e-05)

        return iou
    
    conf_mask = np.expand_dims((predictions_with_boxes[:, :, 4] > confidence_threshold), -1)
    predictions = predictions_with_boxes * conf_mask

    result = {}
    for i, image_pred in enumerate(predictions):
        shape = image_pred.shape
        non_zero_idxs = np.nonzero(image_pred)
        image_pred = image_pred[non_zero_idxs]
        image_pred = image_pred.reshape(-1, shape[-1])
        bbox_attrs = image_pred[:, :5]
        classes = image_pred[:, 5:]
        classes = np.argmax(classes, axis=-1)
    
        unique_classes = list(set(classes.reshape(-1)))

        for cls in unique_classes:
            cls_mask = classes == cls
            cls_boxes = bbox_attrs[np.nonzero(cls_mask)]
            cls_boxes = cls_boxes[cls_boxes[:, -1].argsort()[::-1]]
            cls_scores = cls_boxes[:, -1]
            cls_boxes = cls_boxes[:, :-1]

            while len(cls_boxes) > 0:
                box = cls_boxes[0]
                score = cls_scores[0]
                if not cls in result:
                    result[cls] = []
                result[cls].append((box, score))
                cls_boxes = cls_boxes[1:]
                ious = np.array([iou(box, x) for x in cls_boxes])
                iou_mask = ious < iou_threshold
                cls_boxes = cls_boxes[np.nonzero(iou_mask)]
                cls_scores = cls_scores[np.nonzero(iou_mask)]

    return result

def main():
    '''
    interpreter = tf.lite.Interpreter(model_path=mod_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("{}{}".format(input_details, output_details))

    input_shape = input_details[0]['shape']
    input_data, orig_img = process_image(img_path, input_shape[1], input_shape[2])
    print("{}".format(input_data.shape))
    input_data = input_data.astype(np.float32).reshape((1,416,416,3))
    
    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    detections = convert_box_coordinates(output_data)
    print(detections)
    
    filtered_boxes = non_max_suppression(detections, confidence_threshold=0.1, iou_threshold=0)

    print(filtered_boxes)

    '''
    input_data, orig_img = process_image(img_path, 416, 416)
    input_data = input_data.astype(np.float32).reshape((1,416,416,3))
    
    new_model = tf.keras.models.load_model('model')
    prediction = new_model.predict(input_data)

    print(prediction)
    

if __name__ == "__main__":
    main()
