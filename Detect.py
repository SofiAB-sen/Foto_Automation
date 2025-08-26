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
from io import BytesIO
import requests


import func

#Models
characters_model = YOLO("yolo11n_OCR_v2.pt") 

parser = argparse.ArgumentParser(description='Detect plates and compare with given information.')
parser.add_argument('--process_id', required=True, help='Process ID')
parser.add_argument('--image_path', required=True, help='Path to image file or folder')
parser.add_argument('--detected_plate', required=True, help='Primary plate detection')

args = parser.parse_args()

#Obtener argumentos 
process_id = args.process_id
image_path = args.image_path.replace("\\", "")
detected_plate = args.detected_plate


def process_image(path, process_id):
    image_pil = func.load_image(path, process_id)
    if image_pil is None:
        func.send_log(process_id, "Could not download image.", status="Revision Needed")
        return
    result = func.run_yolo_detector_characteres(characters_model, image_pil)
    if result:
        plate = result[0]
        confidence = result[1]
        print(f"Detected plate: {plate}, Mean confidence: {confidence:.2f}")
        if confidence < 0.7:
            func.send_log(process_id, "Low confidence in plate detection, human revision needed.", status="Revision Needed")
            return 
        
        # Compare with the detected plate
        if plate == detected_plate:
            func.send_log(process_id, f"Plate {plate} matches the camera detected plate {detected_plate}.", status="Match")
        else:

            func.send_log(process_id, f"Plate {plate} does not match the camera detected plate {detected_plate}.", status="Mismatch")
    else:
        func.send_log(process_id, "No detection.", status="Revision Needed")

process_image(image_path, process_id)


