from datetime import datetime
from ultralytics import YOLO
import torch
import os
import numpy as np
from PIL import Image
from ultralytics import YOLO
import cv2
import glob 
import requests
from io import BytesIO

##Function detection of characters in the plate
def run_yolo_detector_characteres(model, image_pil):
    results = model(image_pil)  
    detections = results[0].boxes
    top_boxes = sorted(detections, key=lambda box: float(box.conf), reverse=True)[:6]

    sorted_boxes = sorted(top_boxes, key=lambda box: float(box.xyxy[0][0]))

    plate = []
    confidences = []
    for box in sorted_boxes:
        cls = int(box.cls)
        label = results[0].names[cls]
        plate.append(label)
        conf = float(box.conf)
        confidences.append(conf)
        
        #print(f"🔎 Detected '{label}' with {conf:.2f} confidence at [{x1}, {y1}, {x2}, {y2}]")

    plate = ''.join(plate)
    mean_confidence = np.mean(confidences) if confidences else 0.0
    #print(f"Mean confidence: {mean_confidence:.2f}")
    #print(f"plate: {plate}")
    return plate, mean_confidence

##Function to send logs 
def send_log(process_id, message, status, endpoint, result=False):
    url = endpoint
    payload = {
        "process_id": process_id,
        "status": status,
        "result": result,
        "message": message,
        "date_time": datetime.now().isoformat()
    }
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
    except Exception as e:
        print(f"Failed to send log: {e}")

##Function to load image from path or URL
def load_image(path, process_id, endpoint_url):
    if path.startswith("http://") or path.startswith("https://"):
        try:
            response = requests.get(path, timeout=30)
            response.raise_for_status()
            return Image.open(BytesIO(response.content)).convert("RGB")
        except Exception as e:
            send_log(process_id, f"Error downloading from URL: {e}", status="Revision Needed", endpoint=endpoint_url)
            return None
    elif os.path.exists(path):
        return Image.open(path).convert("RGB")
    else:
        send_log(process_id, "The path does not exist or is not valid.", status="Revision Needed", endpoint=endpoint_url)
        return None 