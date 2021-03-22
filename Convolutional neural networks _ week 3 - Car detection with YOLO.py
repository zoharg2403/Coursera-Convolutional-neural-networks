import numpy as np
import tensorflow as tf
from keras import backend as K

#######################
#  yolo_filter_boxes  #
#######################

def yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold=.6):
    """
    Filters YOLO boxes by thresholding on object and class confidence.
    Arguments:
            box_confidence -- tensor of shape (19, 19, 5, 1)
            boxes -- tensor of shape (19, 19, 5, 4)
            box_class_probs -- tensor of shape (19, 19, 5, 80)
            threshold -- real value, if [ highest class probability score < threshold], then get rid of the corresponding box
    Returns:
            scores -- tensor of shape (None,), containing the class probability score for selected boxes
            boxes -- tensor of shape (None, 4), containing (b_x, b_y, b_h, b_w) coordinates of selected boxes
            classes -- tensor of shape (None,), containing the index of the class detected by the selected boxes
    """

    # Step 1: Compute box scores
    box_scores = box_confidence * box_class_probs

    # Step 2: Find the box_classes using the max box_scores, keep track of the corresponding score
    box_classes = K.argmax(box_scores, axis=-1)
    box_class_scores = K.max(box_scores, axis=-1, keepdims=False)

    # Step 3: Create a filtering mask based on "box_class_scores" by using "threshold"
    filtering_mask = (box_class_scores >= threshold)

    # Step 4: Apply the mask to box_class_scores, boxes and box_classes
    scores = tf.boolean_mask(box_class_scores, filtering_mask)
    boxes = tf.boolean_mask(boxes, filtering_mask)
    classes = tf.boolean_mask(box_classes, filtering_mask)

    return scores, boxes, classes


#########
#  iou  #
#########

def iou(box1, box2):
    """
    Implement the intersection over union (IoU) between box1 and box2
    Arguments:
            box1 -- first box, list object with coordinates (box1_x1, box1_y1, box1_x2, box_1_y2)
            box2 -- second box, list object with coordinates (box2_x1, box2_y1, box2_x2, box2_y2)
    """

    # Assign variable names to coordinates for clarity
    (box1_x1, box1_y1, box1_x2, box1_y2) = box1
    (box2_x1, box2_y1, box2_x2, box2_y2) = box2

    # Calculate the (yi1, xi1, yi2, xi2) coordinates of the intersection of box1 and box2. Calculate its Area.
    xi1 = np.maximum(box1[0], box2[0])
    yi1 = np.maximum(box1[1], box2[1])
    xi2 = np.minimum(box1[2], box2[2])
    yi2 = np.minimum(box1[3], box2[3])
    inter_width = xi2 - xi1
    inter_height = yi2 - yi1
    inter_area = max(inter_width, 0) * max(inter_height, 0)  # Case in which they don't intersec --> max(,0)

    # Calculate the Union area by using Formula: Union(A,B) = A + B - Inter(A,B)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    # compute the IoU
    iou = float(inter_area) / float(union_area)

    return iou

##############################
#  YOLO non-max suppression  #
##############################

def yolo_non_max_suppression(scores, boxes, classes, max_boxes=10, iou_threshold=0.5):
    """
    Applies Non-max suppression (NMS) to set of boxes
    Arguments:
            scores -- tensor of shape (None,), output of yolo_filter_boxes()
            boxes -- tensor of shape (None, 4), output of yolo_filter_boxes() that have been scaled to the image size (see later)
            classes -- tensor of shape (None,), output of yolo_filter_boxes()
            max_boxes -- integer, maximum number of predicted boxes you'd like
            iou_threshold -- real value, "intersection over union" threshold used for NMS filtering
    Returns:
            scores -- tensor of shape (, None), predicted score for each box
            boxes -- tensor of shape (4, None), predicted box coordinates
            classes -- tensor of shape (, None), predicted class for each box
    """

    max_boxes_tensor = K.variable(max_boxes, dtype='int32')  # tensor to be used in tf.image.non_max_suppression()
    K.get_session().run(tf.variables_initializer([max_boxes_tensor]))  # initialize variable max_boxes_tensor

    # Get the list of indices corresponding to boxes you keep
    nms_indices = tf.image.non_max_suppression(boxes, scores, max_output_size=max_boxes, iou_threshold=iou_threshold)

    # Select only nms_indices from scores, boxes and classes
    scores = K.gather(scores, nms_indices)
    boxes = K.gather(boxes, nms_indices)
    classes = K.gather(classes, nms_indices)

    return scores, boxes, classes

###############
#  YOLO eval  #
###############

##  yolo boxes to corners  ##

def yolo_boxes_to_corners(box_xy, box_wh):
    """Convert YOLO box predictions to bounding box corners."""
    box_mins = box_xy - (box_wh / 2.)
    box_maxes = box_xy + (box_wh / 2.)

    return K.concatenate([
        box_mins[..., 1:2],  # y_min
        box_mins[..., 0:1],  # x_min
        box_maxes[..., 1:2],  # y_max
        box_maxes[..., 0:1]  # x_max
    ])

##  scale boxes  ##

def scale_boxes(boxes, image_shape):
    """ Scales the predicted boxes in order to be drawable on the image"""
    height = image_shape[0]
    width = image_shape[1]
    image_dims = K.stack([height, width, height, width])
    image_dims = K.reshape(image_dims, [1, 4])
    boxes = boxes * image_dims
    return boxes

##  yolo eval  ##

def yolo_eval(yolo_outputs, image_shape=(720., 1280.), max_boxes=10, score_threshold=.6, iou_threshold=.5):
    """
    Converts the output of YOLO encoding (a lot of boxes) to your predicted boxes along with their scores, box coordinates and classes.
    Arguments:
            yolo_outputs -- output of the encoding model (for image_shape of (608, 608, 3)), contains 4 tensors:
                            box_confidence: tensor of shape (None, 19, 19, 5, 1)
                            box_xy: tensor of shape (None, 19, 19, 5, 2)
                            box_wh: tensor of shape (None, 19, 19, 5, 2)
                            box_class_probs: tensor of shape (None, 19, 19, 5, 80)
            image_shape -- tensor of shape (2,) containing the input shape, in this notebook we use (608., 608.) (has to be float32 dtype)
            max_boxes -- integer, maximum number of predicted boxes you'd like
            score_threshold -- real value, if [ highest class probability score < threshold], then get rid of the corresponding box
            iou_threshold -- real value, "intersection over union" threshold used for NMS filtering
    Returns:
            scores -- tensor of shape (None, ), predicted score for each box
            boxes -- tensor of shape (None, 4), predicted box coordinates
            classes -- tensor of shape (None,), predicted class for each box
    """

    # Retrieve outputs of the YOLO model
    box_confidence, box_xy, box_wh, box_class_probs = yolo_outputs

    # Convert boxes to be ready for filtering functions (convert boxes box_xy and box_wh to corner coordinates)
    boxes = yolo_boxes_to_corners(box_xy, box_wh)

    # Use one of the functions you've implemented to perform Score-filtering with a threshold of score_threshold
    scores, boxes, classes = yolo_filter_boxes(box_confidence, boxes, box_class_probs, score_threshold)

    # Scale boxes back to original image shape.
    boxes = scale_boxes(boxes, image_shape)

    # Perform Non-max suppression
    scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes, max_boxes, iou_threshold)

    return scores, boxes, classes
