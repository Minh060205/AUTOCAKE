
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import random


class CakeClassifyDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.samples = self._make_dataset()

    def _make_dataset(self):
        samples = []
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            for fname in sorted(os.listdir(class_dir)):
                path = os.path.join(class_dir, fname)
                if os.path.isfile(path):
                    samples.append((path, self.class_to_idx[class_name]))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert('RGB')
        except Exception as e:
            img = Image.new('RGB', (128, 128))
            print(f"Warning: can't open {path}: {e}")
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)
        return img, torch.tensor(label, dtype=torch.long)


def get_classify_transforms(image_size=128):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    train_transform = transforms.Compose([
        transforms.Resize((int(image_size*1.06), int(image_size*1.06))),
        transforms.RandomCrop((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15), 
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0)
    ])
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
transforms.Normalize(mean=mean, std=std)
    ])

    return train_transform, val_transform


def make_train_val_datasets(root_dir, val_split=0.2, seed=42, image_size=128):
    full = CakeClassifyDataset(root_dir, transform=None)
    n = len(full)
    indices = list(range(n))
    random.seed(seed)
    random.shuffle(indices)

    val_size = int(n * val_split)
    train_size = n - val_size
    train_idx = indices[:train_size]
    val_idx   = indices[train_size:]

    train_t, val_t = get_classify_transforms(image_size=image_size)

    train_full = CakeClassifyDataset(root_dir, transform=train_t)
    val_full   = CakeClassifyDataset(root_dir, transform=val_t)

    from torch.utils.data import Subset
    train_dataset = Subset(train_full, train_idx)
    val_dataset   = Subset(val_full, val_idx)

    return train_dataset, val_dataset, full.classes
