import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as T
import streamlit as st
from model_classify import SmallCakeNet
from model_detect import BalancedDetector
from price_map import CAKE_PRICES_MAP, CLASSIFY_CLASSES, DISPLAY_NAMES, format_currency

CLASSIFIER_PATH = 'models/cake_classifier.pth'
DETECTOR_PATH = 'models/cake_detector.pth'
GRID_S = 13
NUM_CLASSES_CLASSIFY = len(CLASSIFY_CLASSES)
CLASSIFIER_IMG_SIZE = 128
DETECTOR_IMG_SIZE = 416
CONF_THRESHOLD = 0.4
NMS_THRESHOLD = 0.05
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

@st.cache_resource 
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading models to {device}...")
    detector, unet, classifier = None, None, None

    try:
        detector = BalancedDetector(S=GRID_S, B=1, C=0)
        detector.load_state_dict(torch.load(DETECTOR_PATH, map_location=device))
        detector.to(device).eval()
        print(f" Loaded Detector (BalancedDetector) from {DETECTOR_PATH}")
    except Exception as e:
        st.error(f"Detector load error: {e}")

    unet = None
    print("UNet model is skipped (theo yêu cầu).")

    try:
        classifier = SmallCakeNet(num_classes=NUM_CLASSES_CLASSIFY)
        classifier.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=device))
        classifier.to(device).eval()
        print(f"Loaded Classifier (SmallCakeNet) from {CLASSIFIER_PATH}")
    except Exception as e:
        st.error(f"Classifier load error: {e}")

    return detector, unet, classifier

def preprocess_image(image_pil, target_size):
    transform = T.Compose([
        T.Resize((target_size, target_size)),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    tensor = transform(image_pil).unsqueeze(0)
    return tensor

@torch.no_grad()
def decode_detector_output(prediction, S=13, conf_threshold=0.4):
    prediction = prediction.squeeze(0)
    bboxes = []
    for i in range(S):
        for j in range(S):
            cell_data = prediction[i, j]
            conf = torch.sigmoid(cell_data[0]).item()
            if conf > conf_threshold:
                x_cell = torch.sigmoid(cell_data[1]).item()
                y_cell = torch.sigmoid(cell_data[2]).item()
                w_norm, h_norm = cell_data[3].item(), cell_data[4].item()
                x_center_norm = (j + x_cell) / S
                y_center_norm = (i + y_cell) / S
                if w_norm > 0 and h_norm > 0 and w_norm <= 1 and h_norm <= 1:
                    bboxes.append([x_center_norm, y_center_norm, w_norm, h_norm, conf])
    return bboxes

def compute_iou(box1, box2):
    def to_corners(box):
        x1 = box[0] - box[2] / 2
        y1 = box[1] - box[3] / 2
        x2 = box[0] + box[2] / 2
        y2 = box[1] + box[3] / 2 
        return [x1, y1, x2, y2]
    
    box1_corners = to_corners(box1)
    box2_corners = to_corners(box2)
    x1_inter = max(box1_corners[0], box2_corners[0])
    y1_inter = max(box1_corners[1], box2_corners[1])
    x2_inter = min(box1_corners[2], box2_corners[2])
    y2_inter = min(box1_corners[3], box2_corners[3])
    
    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]
    union_area = area1 + area2 - inter_area + 1e-6
    return inter_area / union_area

def simple_nms(bboxes, iou_threshold=0.5):
    """NMS đơn giản"""
    bboxes = sorted(bboxes, key=lambda b: b[4], reverse=True)
    bboxes_after_nms = []
    while bboxes:
        chosen_box = bboxes.pop(0)
        bboxes_after_nms.append(chosen_box)
        bboxes = [box for box in bboxes if compute_iou(chosen_box, box) < iou_threshold]
    return bboxes_after_nms

@torch.no_grad()
def run_inference_pipeline(detector, unet, classifier, pil_image):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    img_tensor_detect = preprocess_image(pil_image, DETECTOR_IMG_SIZE).to(device)
    prediction = detector(img_tensor_detect)
    
    bboxes_norm = decode_detector_output(prediction, S=GRID_S, conf_threshold=CONF_THRESHOLD)
    bboxes_nms = simple_nms(bboxes_norm, iou_threshold=NMS_THRESHOLD)
    
    original_cv = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    h0, w0 = original_cv.shape[:2]

    if not bboxes_nms:
        return original_cv, 0, []

    total_price = 0
    results = [] 
    
    for box_norm in bboxes_nms:
        
        x_c_norm, y_c_norm, w_norm, h_norm = box_norm[:4]
        x_c, y_c, w, h = x_c_norm * w0, y_c_norm * h0, w_norm * w0, h_norm * h0
        x1, y1 = int(x_c - w / 2), int(y_c - h / 2)
        x2, y2 = int(x_c + w / 2), int(y_c + h / 2)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w0, x2), min(h0, y2)
        
        try:
            crop_pil = pil_image.crop((x1, y1, x2, y2))
            if crop_pil.width == 0 or crop_pil.height == 0:
                continue
        except Exception:
            continue
            
        if unet is not None:
            pass
            input_pil_image = crop_pil 
        else:
            input_pil_image = crop_pil
        
        cls_in = preprocess_image(input_pil_image, CLASSIFIER_IMG_SIZE).to(device)
        cls_pred_logits = classifier(cls_in)
        cls_pred_idx = torch.argmax(cls_pred_logits, dim=1).item()
        
        class_key = CLASSIFY_CLASSES[cls_pred_idx]
        cls_name = DISPLAY_NAMES.get(class_key, "Unknown")
        price = CAKE_PRICES_MAP.get(cls_name, 0)
        
        if price > 0:
            total_price += price
            results.append({"name": cls_name, "price": price})
            cv2.rectangle(original_cv,(x1,y1),(x2,y2),(0,180,80),2)

            cv2.putText(original_cv,f"{cls_name}: {format_currency(price)}",(x1,max(20,y1-5)),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,(50,30,10),2)

    return original_cv, total_price, results