import torch
import numpy as np
import cv2

def load_classifier_model(model_path, num_classes=10, device='cpu'):
    """Tải model classifier đã train"""
    from model_classify import CustomCNNClassifier
    model = CustomCNNClassifier(num_classes=num_classes)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        print(f"Đã tải model Classifier từ {model_path}")
        return model
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file model classifier tại {model_path}")
        return None
    except Exception as e:
        print(f"Lỗi khi tải model classifier: {e}")
        return None
        
def load_detector_model(model_path, S=13, B=1, C=0, device='cpu'):
    """Tải model detector đã train"""
    from model_detect import CustomObjectDetector
    model = CustomObjectDetector(S=S, B=B, C=C)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        print(f"Đã tải model Detector từ {model_path}")
        return model
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file model detector tại {model_path}")
        return None
    except Exception as e:
        print(f"Lỗi khi tải model detector: {e}")
        return None

def preprocess_image_classify(image_pil, device='cpu'):
    """Chuẩn bị ảnh PIL cho model classifier"""
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    tensor = transform(image_pil).unsqueeze(0) # Thêm batch dimension
    return tensor.to(device)

def preprocess_image_detect(image_pil, device='cpu'):
    """Chuẩn bị ảnh PIL cho model detector"""
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((416, 416)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    tensor = transform(image_pil).unsqueeze(0) # Thêm batch dimension
    return tensor.to(device)

def decode_detector_output(prediction, S=13, conf_threshold=0.4):
    """
    Giải mã output (1, S, S, 5) từ model detector
    thành danh sách các bounding boxes.
    """
    prediction = prediction.squeeze(0) # Bỏ batch dim -> (S, S, 5)
    
    bboxes = [] # Danh sách [x_center_norm, y_center_norm, w_norm, h_norm, conf]
    
    for i in range(S): # Y grid
        for j in range(S): # X grid
            cell_data = prediction[i, j] # (5)
            
            conf = torch.sigmoid(cell_data[0]).item()
            
            if conf > conf_threshold:
                # [conf, x, y, w, h]
                
                # Tọa độ (x, y) tương đối với cell (đã qua sigmoid)
                x_cell = torch.sigmoid(cell_data[1]).item()
                y_cell = torch.sigmoid(cell_data[2]).item()
                
                # Kích thước (w, h) (chưa chuẩn hóa)
                w = cell_data[3].item()
                h = cell_data[4].item()
                
                # Chuyển về tọa độ (x_center, y_center) tương đối với toàn ảnh
                x_center_norm = (j + x_cell) / S
                y_center_norm = (i + y_cell) / S
                
                # Kích thước (w, h) (đã chuẩn hóa)
                # Model của ta dự đoán w, h trực tiếp
                w_norm = w 
                h_norm = h
                
                bboxes.append([x_center_norm, y_center_norm, w_norm, h_norm, conf])
                
    return bboxes

def simple_nms(bboxes, iou_threshold=0.5):
    """
    Non-Maximum Suppression (NMS) đơn giản.
    bboxes: list of [x_center, y_center, w, h, conf]
    """
    
    # Sắp xếp bboxes theo confidence giảm dần
    bboxes = sorted(bboxes, key=lambda b: b[4], reverse=True)
    
    bboxes_after_nms = []
    
    while bboxes:
        # Lấy box có conf cao nhất
        chosen_box = bboxes.pop(0)
        bboxes_after_nms.append(chosen_box)
        
        # So sánh box này với các box còn lại
        bboxes = [
            box for box in bboxes
            if calculate_iou(chosen_box, box) < iou_threshold
        ]
        
    return bboxes_after_nms

def calculate_iou(box1, box2):
    """
    Tính Intersection over Union (IoU)
    box: [x_center, y_center, w, h, conf]
    """
    # Chuyển (x_c, y_c, w, h) -> (x1, y1, x2, y2)
    def to_corners(box):
        x1 = box[0] - box[2] / 2
        y1 = box[1] - box[3] / 2
        x2 = box[0] + box[2] / 2
        y2 = box[1] + box[3] / 2
        return [x1, y1, x2, y2]
        
    box1_corners = to_corners(box1)
    box2_corners = to_corners(box2)
    
    # Tọa độ của vùng giao (intersection)
    x1_inter = max(box1_corners[0], box2_corners[0])
    y1_inter = max(box1_corners[1], box2_corners[1])
    x2_inter = min(box1_corners[2], box2_corners[2])
    y2_inter = min(box1_corners[3], box2_corners[3])
    
    # Diện tích giao
    inter_width = max(0, x2_inter - x1_inter)
    inter_height = max(0, y2_inter - y1_inter)
    inter_area = inter_width * inter_height
    
    # Diện tích của 2 box
    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]
    
    # Diện tích hợp (union)
    union_area = area1 + area2 - inter_area
    
    if union_area == 0:
        return 0
        
    iou = inter_area / union_area
    return iou

def draw_boxes(image_pil, bboxes_with_class):
    """
    Vẽ bounding boxes và tên class lên ảnh (dùng OpenCV).
    bboxes_with_class: list of [x1, y1, x2, y2, class_name, price]
    """
    img_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    
    for box in bboxes_with_class:
        x1, y1, x2, y2, class_name, price = box
        
        # Vẽ box (màu xanh lá)
        cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Chuẩn bị text
        text = f"{class_name}: {price} VND"
        
        # Lấy kích thước text
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        
        # Vẽ nền cho text
        cv2.rectangle(img_cv, (x1, y1 - text_height - baseline), (x1 + text_width, y1), (0, 255, 0), -1)
        
        # Vẽ text (màu đen)
        cv2.putText(img_cv, text, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)