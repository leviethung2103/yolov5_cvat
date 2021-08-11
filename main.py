import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, letterbox
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
import json
import base64
import io
from PIL import Image


CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
WEIGHT_PATH = "yolov5s.pt"
IMG_SIZE = 416
CONFIG = {}

def convert_PIL_to_numpy(image, format):
    """
    Convert PIL image to numpy array of target format.

    Args:
        image (PIL.Image): a PIL image
        format (str): the format of output image

    Returns:
        (np.ndarray): also see `read_image`
    """
    if format is not None:
        # PIL only supports RGB, so convert to RGB and flip channels over below
        conversion_format = format
        if format in ["BGR", "YUV-BT.601"]:
            conversion_format = "RGB"
        image = image.convert(conversion_format)
    image = np.asarray(image)
    # PIL squeezes out the channel dimension for "L", so make it HWC
    if format == "L":
        image = np.expand_dims(image, -1)

    # handle formats not supported by PIL
    elif format == "BGR":
        # flip channels if needed
        image = image[:, :, ::-1]
    elif format == "YUV-BT.601":
        image = image / 255.0
        image = np.dot(image, np.array(_M_RGB2YUV).T)

    return image

def init_context(context):
    context.logger.info("Init context...  0%")

    if torch.cuda.is_available():
        device = select_device('0') # hardcode
    else:
        device = select_device('cpu')

    CONFIG["device"] = device

    half = device.type != 'cpu'  # half precision only supported on CUDA

    # ---------------------- Model Initialization  ------------------------------- 
    predictor = attempt_load(WEIGHT_PATH, map_location=device)  # load FP32 model

    CONFIG["stride"] = int(predictor.stride.max())  # model stride

    # Get names and colors
    names = predictor.module.names if hasattr(predictor, 'module') else predictor.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    context.user_data.model_handler = predictor

    context.logger.info("Init context...100%")


def handler(context,event):
    """ Control the threshold by application """

    context.logger.info("Run Yolov5 model - License Plate")
    data = event.body
    buf = io.BytesIO(base64.b64decode(data["image"]))
    threshold = float(data.get("threshold", 0.5))
    # image: BGR format
    img0 = convert_PIL_to_numpy(Image.open(buf), format="BGR")

    # preprocess 
    stride = CONFIG["stride"]  # model stride
    img = letterbox(img0, IMG_SIZE, stride=stride)[0] 

    # Convert       
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(CONFIG["device"])
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    pred = context.user_data.model_handler(img, augment=False)[0]

    # Apply NMS
    pred = non_max_suppression(pred, CONFIDENCE_THRESHOLD, IOU_THRESHOLD, classes=None, agnostic=False)

    results = []
    # Process detections, # detections per image
    for i, det in enumerate(pred):  
        gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

            for *xyxy, conf, cls in reversed(det):
                print (torch.tensor(xyxy).view(1, 4))
                print (torch.tensor(xyxy).tolist())
                box_list = torch.tensor(xyxy).tolist()
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                line = (cls, *xywh)  # label format
                label = f'{names[int(cls)]} {conf:.2f}'

                print (box_list, float(conf),names[int(cls)])
                if float(conf) >= threshold:
                    results.append({
                        "confidence": str(float(conf)),
                        "label": names[int(cls)],
                        "points": box_list,
                        "type": "rectangle",
                    })

    return context.Response(body=json.dumps(results), headers={},
        content_type='application/json', status_code=200)

if __name__ == '__main__':

    input = "/home/hunglv/Downloads/detectron2/1920px-Cat_poster_1.jpg"

    if torch.cuda.is_available():
        # device = select_device('0') # hardcode
        device = select_device('cpu')
    else:
        device = select_device('cpu')

    half = device.type != 'cpu'  # half precision only supported on CUDA

    # ---------------------- Model Initialization  ------------------------------- 
    predictor = attempt_load(WEIGHT_PATH, map_location=device)  # load FP32 model
    stride = int(predictor.stride.max())  # model stride
    # Get names and colors
    names = predictor.module.names if hasattr(predictor, 'module') else predictor.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # ---------------------- Processing the Image ------------------------------- 
    # Read the image and pad the image 
    img0 = cv2.imread(input) # BGR
    img = letterbox(img0, IMG_SIZE, stride=stride)[0] 

    # Convert       
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # ---------------------- Model Inferencing ------------------------------- 
    pred = predictor(img, augment=False)[0]

    # Apply NMS
    pred = non_max_suppression(pred, CONFIDENCE_THRESHOLD, IOU_THRESHOLD, classes=None, agnostic=False)

    # pred: can be an empty list [], contains conf, label, bounding boxes

    # Process detections, # detections per image
    for i, det in enumerate(pred):  
        gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

            # Write results
            for *xyxy, conf, cls in reversed(det):
                print (torch.tensor(xyxy).view(1, 4))
                print (torch.tensor(xyxy).tolist())
                box_list = torch.tensor(xyxy).tolist()
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                line = (cls, *xywh)  # label format
                print("Line:",line)
                label = f'{names[int(cls)]} {conf:.2f}'

                print ("Label:",label)
                print(box_list)
                cv2.rectangle(img0, (int(box_list[0]),int(box_list[1])), (int(box_list[2]),int(box_list[3])), (0,255,0), 1)
                print (box_list, float(conf),names[int(cls)])


