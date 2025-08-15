from ultralytics import YOLO
import torch
import os
import numpy as np
from PIL import Image
from ultralytics import YOLO
import pandas as pd
import cv2
import glob 
from datetime import datetime
import argparse


import func

#Models
plate_model = YOLO("yolo11n_Plate_Recognition_v2.pt") 
characters_model = YOLO("yolo11n_OCR_v2.pt") 

parser = argparse.ArgumentParser(description='Detect plates and compare with given information.')
parser.add_argument('--process_id', required=True, help='Process ID')
parser.add_argument('--image_path', required=True, help='Path to image file or folder')
parser.add_argument('--detected_plate', required=True, help='Primary plate detection')

args = parser.parse_args()

#Obtener argumentos 
process_id = args.process_id
image_path = args.image_path
detected_plate = args.detected_plate


if not os.path.exists(image_path):
    print("Teh path does not exists.")
    exit()
  

def process_image(path, process_id):
    if not os.path.exists(image_path):
        print("The image path does not exist.")
        return
    image_pil = Image.open(path).convert("RGB")
    result = func.run_yolo_detector_characteres(characters_model, image_pil)
    if result:
        plate = result[0]
        confidence = result[1]
        print(f"Detected plate: {plate}, Mean confidence: {confidence:.2f}")
        if confidence < 0.5:
            func.send_log(process_id, "Low confidence in plate detection, human revision needed.", status="Revision Needed")
            return 
        
        # Compare with the detected plate
        if plate == detected_plate:
            func.send_log(process_id, f"Plate {plate} matches the camera detected plate {detected_plate}.", status="Match")
        else:

            func.send_log(process_id, f"Plate {plate} does not match the camera detected plate {detected_plate}.", status="Mismatch")
    else:
        func.send_log(process_id, "No detection.", status="Revision Needed")

if os.path.isfile(image_path):
    process_image(image_path, process_id)


