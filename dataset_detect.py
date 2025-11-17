import torch
import os
import glob
from PIL import Image, ImageFile
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
import torchvision.transforms as T
import random

ImageFile.LOAD_TRUNCATED_IMAGES = True

class CakeDetectDataset(Dataset):
    def __init__(self, root_dir, S=13, B=1, C=0, transform=None, img_size=416, do_augment=False):
        self.img_dir = os.path.join(root_dir, "images")
        self.label_dir = os.path.join(root_dir, "labels")
        self.img_size = img_size
        self.transform = transform 
        self.S = S
        self.B = B
        self.C = C
        self.output_dim = (B * 5 + C)
        
        self.do_augment = do_augment
        
        self.color_jitter = T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)
        self.grayscale = T.RandomGrayscale(p=0.15)

        self.images = sorted(glob.glob(os.path.join(self.img_dir, "*.jpg"))) + \
                      sorted(glob.glob(os.path.join(self.img_dir, "*.png"))) + \
                      sorted(glob.glob(os.path.join(self.img_dir, "*.jpeg")))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = self.images[index]
        label_name = os.path.basename(img_path).rsplit('.', 1)[0] + '.txt'
        label_path = os.path.join(self.label_dir, label_name)

        image = Image.open(img_path).convert("RGB")
        
        boxes = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        boxes.append([x_center, y_center, width, height])
        
        if self.do_augment:
            image = self.color_jitter(image)
            image = self.grayscale(image)
            
            if random.random() < 0.5:
                image = F.hflip(image)
                for i in range(len(boxes)):
                    boxes[i][0] = 1.0 - boxes[i][0]
        if self.transform:
            image = self.transform(image)
        
        target = torch.zeros((self.S, self.S, self.output_dim))
        
        for box in boxes:
            x_center, y_center, width, height = box
            i, j = int(self.S * y_center), int(self.S * x_center)
            x_cell = (self.S * x_center) - j
            y_cell = (self.S * y_center) - i
            
            if target[i, j, 0] == 0:
                target[i, j, 0] = 1
                target[i, j, 1] = x_cell
                target[i, j, 2] = y_cell
                target[i, j, 3] = width
                target[i, j, 4] = height
                
        return image, target
