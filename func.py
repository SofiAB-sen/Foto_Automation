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

##Funcion deteccion de caracteres
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
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        #print(f"🔎 Detected '{label}' with {conf:.2f} confidence at [{x1}, {y1}, {x2}, {y2}]")

    plate = ''.join(plate)
    mean_confidence = np.mean(confidences) if confidences else 0.0
    #print(f"Mean confidence: {mean_confidence:.2f}")
    #print(f"plate: {plate}")
    return plate, mean_confidence

##Funcion deteccion de placas
"""def run_yolo_detector_plate(model_plates, model_characteres, path):
    
    image_pil = Image.open(path).convert("RGB")
    
    results = model_plates(path)  # Inference

    detections = results[0] 
    boxes = detections.boxes.xyxy.cpu().numpy()  
    scores = detections.boxes.conf.cpu().numpy()  
    class_ids = detections.boxes.cls.cpu().numpy()  
    names = model_plates.names  

    
    min_score = 0.5  # Minimum score threshold for detection
    
    dic = {'plate': [], 'confidence': []}
    for i, box in enumerate(boxes):
        score = scores[i]
        class_id = int(class_ids[i])
        class_name = names[class_id] 

        if score >= min_score and class_name.lower() in ["license plate", "vehicle registration plate", "plate"]:
            print(f"Plate found with {int(100 * score)}% confidence.")

            xmin, ymin, xmax, ymax = map(int, box)
            cropped_image = image_pil.crop((xmin, ymin, xmax, ymax))
            cropped_image = cropped_image.resize((96, 48))
            plate, confidence  = run_yolo_detector_characteres(model_characteres, cropped_image)

            if plate:
                dic['plate'].append(plate)
                dic['confidence'].append(confidence)
                print(f"Plate: {plate}, Confidence: {confidence:.2f}")
    if not dic['plate']:
        print("No plate detected.")
    return dic if dic['plate'] else None"""
  
##Funcion para enviar logs 
def send_log(process_id, message, status):
    url = "https://webhook.site/c016b479-3691-436a-bd36-bcd1c6ead397"
    payload = {
        "process_id": process_id,
        "status": status,
        "message": message,
        "date_time": datetime.now().isoformat()
    }
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
    except Exception as e:
        print(f"Failed to send log: {e}")

##Funcion para cargar imagen desde URL o path local
def load_image(path, process_id):
    if path.startswith("http://") or path.startswith("https://"):
        try:
            response = requests.get(path, timeout=30)
            response.raise_for_status()
            return Image.open(BytesIO(response.content)).convert("RGB")
        except Exception as e:
            send_log(process_id, f"Error downloading from URL: {e}", status="Revision Needed")
            return None
    elif os.path.exists(path):
        return Image.open(path).convert("RGB")
    else:
        send_log(process_id, "The path does not exist or is not valid.", status="Revision Needed")
        return None 