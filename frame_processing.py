import cv2
import numpy as np
import tensorflow as tf
from tflite_models import TensorFlowModel
import os
import time


model = TensorFlowModel()
model.load(os.path.join(os.getcwd(), 'assets/model.tflite'))

net = cv2.dnn.readNet("/home/danny/CapstoneSpring2025/assets/yolov3-tiny.weights","/home/danny/CapstoneSpring2025/assets/yolov3-tiny.cfg")
# net = cv2.dnn.readNet("/home/danny/darknet/yolov3.weights","/home/danny/darknet/cfg/yolov3.cfg")

classes = []
with open('/home/danny/darknet/data/coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

boxes = []

face = False

def pad_bbox(bbox, pad_ratio, frame_width, frame_height):
    """
    Pad a bounding box [x, y, w, h] by pad_ratio while ensuring it stays within frame dimensions.
    """
    x, y, w, h = bbox
    pad_w = int(4 * w * pad_ratio)
    pad_h = int(h * pad_ratio)
    new_x = max(x - pad_w // 2, 0)
    new_y = max(y - pad_h // 2, 0)
    new_w = min(w + pad_w, frame_width - new_x)
    new_h = min(h + pad_h, frame_height - new_y)
    return [new_x, new_y, new_w, new_h]


def process_yolo(img, pad_ratio=0.2,):
    height, width, channels = img.shape

    blob = cv2.dnn.blobFromImage(img, 1/255.0, (416,416), swapRB=True, crop=False)
    net.setInput(blob) 
    layer_outputs = net.forward(output_layers)

    boxes = []
    confidences = []
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if classes[classID] == "person" and confidence > 0.5:
                box = detection[0:4] * np.array([width, height, width, height])
                (centerX, centerY, width_box, height_box) = box.astype("int")
                x_box = int(centerX - width_box / 2)
                y_box = int(centerY - height_box / 2)
                boxes.append([x_box, y_box, int(width_box), int(height_box)])
                confidences.append(float(confidence))
    
    # get indices
    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.3)

    # if indices greater than 0, crop the image, otherwise return none
    last_bbox = None
    if len(indices) > 0:
        indices = np.array(indices).flatten()
        best_index = indices[0]
        bbox = boxes[best_index]
        padded_bbox = pad_bbox(bbox, pad_ratio, frame_width=width, frame_height=height)
        last_bbox = padded_bbox
        (px, py, pw, ph) = last_bbox
        # crop = frame[py:py+ph, px:px+pw]
    return last_bbox


def movenet_preprocessing(image):
    # Load the input image.
    image = tf.convert_to_tensor(image)
    input_size = 192
    # Resize and pad the image to keep the aspect ratio and fit the expected size.
    input_image = tf.expand_dims(image, axis=0)
    input_image = tf.image.resize_with_pad(input_image, input_size, input_size)
    # SavedModel format expects tensor type of int32.
    input_image = tf.cast(input_image, dtype=tf.int32)
    return input_image

def process_movenet_cropped(frame, bbox):
    global model
    frame = movenet_preprocessing(frame)
    res = model.pred(frame)
    res = movenet_postprocessing(res, bbox)
    return res
        # if crop.size != 0:
        #     t_movenet_update = time.time()
        #     last_keypoints = run_movenet_on_crop(crop, movenet)

def movenet_postprocessing(kps, bbox):
    if kps is not None and len(kps) is not 0:
        
        return keypoints_to_pixels(kps, bbox)
def check_face(kps):
    """Determines which way the person is facing, either towards,
    or away from the camera"""
    left_shoulder = kps[5][1]
    right_shoulder = kps[6][1]    
    return left_shoulder > right_shoulder


def keypoints_to_pixels(kps, bbox):
    # get x, y of 
    x, y, w, h = bbox

    # When MoveNet resizes the crop to 192x192, padding is added.
    scale = min(192 / h, 192 / w)
    new_h = h * scale
    new_w = w * scale
    pad_top = (192 - new_h) / 2.0
    pad_left = (192 - new_w) / 2.0

    kps = kps[0,0,:,:]

    # Convert from 192x192 padded coordinates back to crop coordinates.
    # do x
    kps[:, 1] = x + (kps[:, 1]*192 - pad_left) / scale
    kps[:, 0] = y + (kps[:, 0]*192 - pad_top) / scale

    # print(f'{kps[5,:2]}\t{kps[6,:2]}')
    # kps[:, 0]
    # x_in_crop = (kx * 192 - pad_left) / scale
    # y_in_crop = (ky * 192 - pad_top) / scale
    # # Map to frame coordinates.
    # point = (int(x + x_in_crop), int(y + y_in_crop))
    return kps

def hips_visible(kps, frame, kp_thresh=0.4):
    thrsh = kps[11,2] >= kp_thresh and kps[12,2] >= kp_thresh
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]
    
    x_in_frame = kps[11,1] <= frame_width and kps[12,1] <= frame_height
    y_in_frame = kps[11,0] <= frame_height and kps[12,0] <= frame_height

    return thrsh and x_in_frame and y_in_frame

def draw_keypoints_and_bbox(frame, bbox, keypoints, kp_thresh=0.2):
    """
    Draw the padded bounding box, keypoints as blue circles, and red lines connecting keypoints.
    The keypoints are transformed from the 192x192 padded space back to the crop coordinates.
    """
    
    x, y, w, h = bbox
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(frame, f"{x},{y},{w},{h}", (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    if keypoints is not None:
    # Normalize keypoints shape to (17, 3)
        if keypoints.ndim == 4:
            keypoints = keypoints[0][0]
        elif keypoints.ndim == 3:
            keypoints = keypoints[0]
        
        # When MoveNet resizes the crop to 192x192, padding is added.
        # scale = min(192 / h, 192 / w)
        # new_h = h * scale
        # new_w = w * scale
        # pad_top = (192 - new_h) / 2.0
        # pad_left = (192 - new_w) / 2.0
        
        points = [None] * len(keypoints)
        for i, kp in enumerate(keypoints):
            kx, ky, score = kp
            kz = kx
            kx = ky
            ky = kz
            if score < kp_thresh:
                continue
            # # Convert from 192x192 padded coordinates back to crop coordinates.
            # x_in_crop = (kx * 192 - pad_left) / scale
            # y_in_crop = (ky * 192 - pad_top) / scale
            # Map to frame coordinates.
            point = (int(kx), int(ky))
            points[i] = point
            cv2.circle(frame, point, 4, (255, 0, 0), -1)
        
        # Define connections (limbs) between keypoints.
        connections = [(0, 1), (0, 2), (1, 3), (2, 4),
                    (0, 5), (0, 6), (5, 7), (7, 9),
                    (6, 8), (8, 10), (5, 6), (5, 11),
                    (6, 12), (11, 12), (11, 13), (13, 15),
                    (12, 14), (14, 16)]
        for (i, j) in connections:
            if i < len(points) and j < len(points) and points[i] is not None and points[j] is not None:
                cv2.line(frame, points[i], points[j], (0, 0, 255), 2)

    return frame

def box_preprocessing(box):
    (x, y, w, h) = box
    return (x, y, x + w, y + h)

def calculate_iou(box1, box2):
    """
    Calculates the Intersection over Union (IoU) of two bounding boxes.

    Args:
        box1 (list or tuple): Bounding box in the format (x1, y1, x2, y2).
        box2 (list or tuple): Bounding box in the format (x1, y1, x2, y2).

    Returns:
        float: IoU score.
    """
    x1_intersect = max(box1[0], box2[0])
    y1_intersect = max(box1[1], box2[1])
    x2_intersect = min(box1[2], box2[2])
    y2_intersect = min(box1[3], box2[3])

    intersection_area = max(0, x2_intersect - x1_intersect) * max(0, y2_intersect - y1_intersect)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area

    if union_area == 0:
        return 0

    iou = intersection_area / union_area
    return iou

def valid_box(boxA, boxB, thresh=0.):
    """Determines if boxA is a valid next box in comparison with boxB"""
    boxa = box_preprocessing(boxA)
    boxb = box_preprocessing(boxB)
    iou = calculate_iou(boxa, boxb)
    print(iou)
    return iou