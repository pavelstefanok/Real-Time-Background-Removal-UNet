import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import os
import numpy as np
import random
from model import SimpleUNet


# ----------------------------
# Dataset
# ----------------------------
class PersonDatasetSubset(Dataset):
    def __init__(self, images_dir, masks_dir, file_pairs, augment=False):
        self.file_pairs = file_pairs  # Lista de (nume_img, nume_mask)
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.augment = augment

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):
        img_name, mask_name = self.file_pairs[idx]
        img = cv2.imread(os.path.join(self.images_dir, img_name))
        mask = cv2.imread(os.path.join(self.masks_dir, mask_name), cv2.IMREAD_GRAYSCALE)

        # Protectie in caz de fisier corupt
        if img is None or mask is None:
            img = np.zeros((256, 256, 3), dtype=np.uint8)
            mask = np.zeros((256, 256), dtype=np.uint8)
        else:
            img = cv2.resize(img, (256, 256))
            mask = cv2.resize(mask, (256, 256))

        if self.augment and random.random() > 0.5:
            img = cv2.flip(img, 1)
            mask = cv2.flip(mask, 1)

        img = img.astype(np.float32) / 255.0
        mask = (mask > 127).astype(np.float32)

        img_tensor = torch.from_numpy(img).permute(2, 0, 1)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)
        return img_tensor, mask_tensor


# ----------------------------
# Config & Split
# ----------------------------
images_dir = "../dataset/images"
masks_dir = "../dataset/masks"
max_images = 12188
batch_size = 8
num_epochs = 12
lr = 1e-3

# REZOLVARE EXTENSII: Mapeaza numele "curat" la "nume.extensie"
img_dict = {os.path.splitext(f)[0]: f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))}
mask_dict = {os.path.splitext(f)[0]: f for f in os.listdir(masks_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))}

# Gaseste doar fisierele care exista in ambele foldere
common_names = sorted(list(set(img_dict.keys()) & set(mask_dict.keys())))[:max_images]
all_pairs = [(img_dict[name], mask_dict[name]) for name in common_names]

random.seed(42)
random.shuffle(all_pairs)

train_idx = int(0.8 * len(all_pairs))
val_idx = int(0.9 * len(all_pairs))

train_files = all_pairs[:train_idx]
val_files = all_pairs[train_idx:val_idx]
test_files = all_pairs[val_idx:]

train_ds = PersonDatasetSubset(images_dir, masks_dir, train_files, augment=True)
val_ds = PersonDatasetSubset(images_dir, masks_dir, val_files, augment=False)
test_ds = PersonDatasetSubset(images_dir, masks_dir, test_files, augment=False)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleUNet().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

#pentru prezentare
train_ious = []
val_ious = []

def compute_iou(pred, target, threshold=0.5):
    pred_bin = (pred > threshold).float()
    intersection = (pred_bin * target).sum()
    union = pred_bin.sum() + target.sum() - intersection
    return (intersection / (union + 1e-6)).item()


# ----------------------------
# Training Loop
# ----------------------------
print(f"\nIncepe antrenarea pe {device}...")
print(f"Perechi gasite: {len(all_pairs)}")
best_val_iou = 0.0

for epoch in range(num_epochs):
    model.train()
    train_loss, train_iou = 0, 0

    if epoch == 20:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 1e-4
        print("Viteza de invatare a scazut la 1e-4 pentru rafinament.")

    for i, (imgs, masks) in enumerate(train_loader):
        imgs, masks = imgs.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        batch_loss = loss.item()
        batch_iou = compute_iou(outputs, masks)
        train_loss += batch_loss
        train_iou += batch_iou

        if (i + 1) % 20 == 0 or (i + 1) == len(train_loader):
            print(f"Epoca [{epoch + 1}/{num_epochs}] | Batch [{i + 1}/{len(train_loader)}] | "
                  f"Loss: {batch_loss:.4f} | IoU: {batch_iou * 100:.1f}%")

    model.eval()
    val_loss, val_iou = 0, 0
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            outputs = model(imgs)
            v_loss = criterion(outputs, masks)
            val_loss += v_loss.item()
            val_iou += compute_iou(outputs, masks)

    avg_train_iou = (train_iou / len(train_loader)) * 100
    avg_val_iou = (val_iou / len(val_loader)) * 100

    print("-" * 50)
    print(f"FINAL EPOCA {epoch + 1}")

    #pt prezentare
    train_ious.append(avg_train_iou)
    val_ious.append(avg_val_iou)

    print(f"Antrenare - IoU: {avg_train_iou:.2f}%")
    print(f"Validare  - IoU: {avg_val_iou:.2f}%")

    if (val_iou / len(val_loader)) > best_val_iou:
        best_val_iou = val_iou / len(val_loader)
        torch.save(model.state_dict(), "best_model.pth")
        print(f"*** BEST MODEL salvat: {avg_val_iou:.2f}% ***")
    print("-" * 50)

# ----------------------------
# pt prezentare
# ----------------------------
print("\n" + "="*30)
print("DATE FINALE pe epoci:")
print(f"train_ious = {train_ious}")
print(f"val_ious = {val_ious}")
print("="*30 + "\n")


# ----------------------------
# Examen
# ----------------------------
print(f"TEST FINAL")
if os.path.exists("best_model.pth"):
    model.load_state_dict(torch.load("best_model.pth"))
model.eval()
test_iou_sum = 0
with torch.no_grad():
    for imgs, masks in test_loader:
        imgs, masks = imgs.to(device), masks.to(device)
        outputs = model(imgs)
        test_iou_sum += compute_iou(outputs, masks)

print(f"IOU FINAL PE TEST: {(test_iou_sum/len(test_loader))*100:.2f}%")