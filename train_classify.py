import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from dataset_classify import make_train_val_datasets
from model_classify import SmallCakeNet

DATA_DIR = 'dataset_classify'
MODEL_SAVE_PATH = 'models/cake_classifier.pth'
NUM_CLASSES = 10
BATCH_SIZE = 32
NUM_EPOCHS = 200
LR = 1e-3
WEIGHT_DECAY = 1e-5
VAL_SPLIT = 0.2
IMAGE_SIZE = 128
SEED = 42
EARLY_STOPPING_PATIENCE = 15


def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = np.inf
        self.counter = 0
        self.best_state = None

    def step(self, loss, model):
        if loss + self.min_delta < self.best_loss:
            self.best_loss = loss
            self.counter = 0
            self.best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            return True
        else:
            self.counter += 1
            return False


def train():
    os.makedirs('models', exist_ok=True)
    set_seed(SEED)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    train_dataset, val_dataset, class_names = make_train_val_datasets(
        DATA_DIR, val_split=VAL_SPLIT, seed=SEED, image_size=IMAGE_SIZE
    )
    print('Found classes:', class_names)
    print('Train samples:', len(train_dataset), 'Val samples:', len(val_dataset))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                            num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                            num_workers=NUM_WORKERS, pin_memory=True)

    model = SmallCakeNet(num_classes=NUM_CLASSES).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    # Dùng scaler cho mixed precision
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

    early = EarlyStopping(patience=EARLY_STOPPING_PATIENCE)

    best_val_acc = 0.0

    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc='Train')
        for inputs, labels in pbar:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            if scaler is not None:
                with torch.amp.autocast(device_type='cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data).item()
            total += inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc='Val'):
                inputs = inputs.to(device)
                labels = labels.to(device)
                if scaler is not None:
                    with torch.amp.autocast(device_type='cuda'):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += torch.sum(preds == labels.data).item()
                val_total += inputs.size(0)

        val_epoch_loss = val_loss / len(val_loader.dataset)
        val_epoch_acc = val_correct / val_total
        print(f"Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}")

        improved = early.step(val_epoch_loss, model)
        if improved:
            torch.save(early.best_state, MODEL_SAVE_PATH)
            best_val_acc = val_epoch_acc
            print(f"Saved best model to {MODEL_SAVE_PATH} (val loss {early.best_loss:.4f})")
        else:
            print(f"No improvement. Early stop counter: {early.counter}/{early.patience}")

        scheduler.step()

        if early.counter >= early.patience:
            print('Early stopping triggered')
            break

    print('Training finished')
    print('Best val loss:', early.best_loss)
    print('Best val acc:', best_val_acc)


if __name__ == '__main__':
    train()
