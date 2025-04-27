import cv2
import numpy as np

net = cv2.dnn.readNet("/home/danny/BrilliantAICapstone/assets/yolov3-tiny.weights","/home/danny/BrilliantAICapstone/assets/yolov3-tiny.cfg")
# net = cv2.dnn.readNet("/home/danny/darknet/yolov3.weights","/home/danny/darknet/cfg/yolov3.cfg")

classes = []
with open('/home/danny/darknet/data/coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

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
    print("5")
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
    print("6")
    return last_bbox

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

def valid_box(boxA, boxB):
    """Determines if boxA is a valid next box in comparison with boxB"""
    boxa = box_preprocessing(boxA)
    boxb = box_preprocessing(boxB)
    iou = calculate_iou(boxa, boxb)
    return iou