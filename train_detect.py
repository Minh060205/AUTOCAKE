import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import numpy as np
from torchvision import transforms

from model_detect import BalancedDetector
from dataset_detect import CakeDetectDataset

DATA_DIR_ROOT = 'dataset_detect'
MODEL_SAVE_PATH = 'models/cake_detector.pth'
GRID_S = 13
NUM_B = 1
NUM_C = 0
BATCH_SIZE = 16
NUM_EPOCHS = 300
LEARNING_RATE = 3e-4 
WEIGHT_DECAY = 1e-3 

LAMBDA_COORD = 5
LAMBDA_NOOBJ = 0.5
EARLY_STOPPING_PATIENCE = 40 
NUM_WORKERS = 4
LABEL_SMOOTHING = 0.1 

class SimpleYOLOLoss(nn.Module):
    def __init__(self, S=13, B=1, C=0, lambda_coord=5.0, lambda_noobj=0.5, label_smoothing=0.0):
        super().__init__()
        self.S = S
        self.B = B
        self.C = C
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.label_smoothing = label_smoothing 
        self.mse = nn.MSELoss(reduction='sum')
        self.bce_logits = nn.BCEWithLogitsLoss(reduction='sum')
        print(f"Loss function initialized with: S={S}, B={B}, C={C}, LS={label_smoothing}")

    def forward(self, pred, target):
        obj_mask = target[..., 0] == 1
        noobj_mask = target[..., 0] == 0

        pred_xy = torch.sigmoid(pred[..., 1:3])
        target_xy = target[..., 1:3]
        loss_xy = self.mse(pred_xy[obj_mask], target_xy[obj_mask])

        pred_wh = pred[..., 3:5]
        target_wh_sqrt = torch.sqrt(target[..., 3:5])
        pred_wh_sqrt = torch.sign(pred_wh) * torch.sqrt(torch.abs(pred_wh) + 1e-6)
        loss_wh = self.mse(pred_wh_sqrt[obj_mask], target_wh_sqrt[obj_mask])

        coord_loss = self.lambda_coord * (loss_xy + loss_wh)

        pred_conf_logits_obj = pred[..., 0][obj_mask]
        target_conf_obj = torch.full_like(pred_conf_logits_obj, (1.0 - self.label_smoothing))
        obj_conf_loss = self.bce_logits(pred_conf_logits_obj, target_conf_obj)

        pred_conf_logits_noobj = pred[..., 0][noobj_mask]
        target_conf_noobj = target[..., 0][noobj_mask]
        noobj_conf_loss = self.bce_logits(pred_conf_logits_noobj, target_conf_noobj)

        conf_loss = obj_conf_loss + (self.lambda_noobj * noobj_conf_loss)

        total_loss = (coord_loss + conf_loss) / pred.size(0)
        return total_loss

def train_detector():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    os.makedirs('models', exist_ok=True)
    train_dir = os.path.join(DATA_DIR_ROOT, 'train')
    val_dir = os.path.join(DATA_DIR_ROOT, 'valid')

    train_transform = transforms.Compose([
        transforms.Resize((416, 416)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0)
    ])

    val_transform = transforms.Compose([
        transforms.Resize((416, 416)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        print(f"Lỗi: Không tìm thấy thư mục 'train' hoặc 'valid' trong '{DATA_DIR_ROOT}'")
        return

    train_dataset = CakeDetectDataset(train_dir, S=GRID_S,
                                      transform=train_transform,
                                      do_augment=True)

    val_dataset = CakeDetectDataset(val_dir, S=GRID_S,
                                    transform=val_transform,
                                    do_augment=False)

    print(f"Tìm thấy {len(train_dataset)} ảnh train (Augment: Jitter, Flip, Grayscale, Cutout)")
    print(f"Tìm thấy {len(val_dataset)} ảnh val (Không Augment)")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    model = BalancedDetector(S=GRID_S, B=NUM_B, C=NUM_C).to(device)

    criterion = SimpleYOLOLoss(S=GRID_S, B=NUM_B, C=NUM_C,
                             lambda_coord=LAMBDA_COORD,
                             lambda_noobj=LAMBDA_NOOBJ,
                             label_smoothing=LABEL_SMOOTHING) 

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    best_val_loss = np.inf
    epochs_no_improve = 0

    scaler = torch.amp.GradScaler() if device.type == 'cuda' else None

    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        model.train()
        running_loss = 0
        pbar = tqdm(train_loader, desc="Train")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            if scaler is not None:
                with torch.amp.autocast(device_type=device.type):
                    pred = model(x)
                    loss = criterion(pred, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                pred = model(x)
                loss = criterion(pred, y)
                loss.backward()
                optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = running_loss / len(train_loader)
        print(f"Train Loss: {train_loss:.4f}")

        model.eval()
        val_loss_total = 0
        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc="Val")
            for x, y in pbar_val:
                x, y = x.to(device), y.to(device)

                if scaler is not None:
                    with torch.amp.autocast(device_type=device.type):
                        pred = model(x)
                        loss = criterion(pred, y)
                else:
                    pred = model(x)
                    loss = criterion(pred, y)

                val_loss_total += loss.item()
                pbar_val.set_postfix(loss=f"{loss.item():.4f}")

        val_loss = val_loss_total / len(val_loader)
        print(f"Val Loss: {val_loss:.4f}")

        scheduler.step() 

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH) 
            print(f"Saved best model to {MODEL_SAVE_PATH} (val loss {best_val_loss:.4f})")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"Val loss không cải thiện. Patience: {epochs_no_improve}/{EARLY_STOPPING_PATIENCE}")
            if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print("Training complete!")

if __name__ == '__main__':
    train_detector()