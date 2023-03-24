## Input: Full Frame Images without resizing
import sys
sys.path.insert(0, "yolov5_cvat")
import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import  non_max_suppression,scale_coords, xyxy2xywh
    
from utils.torch_utils import select_device
import os 


CONFIDENCE_THRESHOLD = 0.30
IOU_THRESHOLD = 0.60
WEIGHT_PATH = "yolov5s.pt"
IMG_SIZE = 640 # or 416

def write_one_line(bbox, class_id, image_width, image_height):
    bbox_x_min, bbox_y_min, bbox_x_max, bbox_y_max = bbox
    bbox_x, bbox_y, bbox_w, bbox_h = bbox_x_min, bbox_y_min, bbox_x_max - bbox_x_min, bbox_y_max - bbox_y_min
    center_x = bbox_x + bbox_w / 2
    center_y = bbox_y + bbox_h / 2
    normalized_center_x = center_x / image_width
    normalized_center_y = center_y / image_height
    normalized_bbox_w = bbox_w / image_width
    normalized_bbox_h = bbox_h / image_height
    return f"{class_id} {normalized_center_x:.6f} {normalized_center_y:.6f} {normalized_bbox_w:.6f} {normalized_bbox_h:.6f}\n"

def inference_image(path, is_draw=False, write_to_file=False, output_dir = "./"):
    
    basename = os.path.basename(path).split(".")[0]

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
    img0 = cv2.imread(path) # BGR
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
    for _, det in enumerate(pred):
        gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
            
            output_path = os.path.join(output_dir, basename) + ".txt"
            file = open(output_path,"w")
            

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
                
                if write_to_file:
                    file.write(write_one_line(box_list,int(cls), img0.shape[1], img0.shape[0]))
                    
            file.close()
                    

    if is_draw:
        cv2.imshow('dAS',img0)
        cv2.waitKey(0)

def inference_folder(img_dir):
    for path in os.listdir(img_dir):
        inference_image(os.path.join(img_dir, path), write_to_file=True)
    
if __name__ == '__main__':

    img_path = "/home/hunglv/Downloads/YOLO_DOCKER/yolov8/yolov5/LPD_LAO_ChienKhuong_21-03-2023_v1.0/test/images/0cc9f35f-35aa-498c-bf10-39b7f59e9449.jpg"
    folder_path = "/media/hunglv/T7/Projects/test"
    
    inference_image(img_path, write_to_file=True)
    inference_folder(folder_path)
    
    