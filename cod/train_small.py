import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import cv2
import os
import numpy as np
import random
from model import SimpleUNet


#  LOSS
class DiceBCELoss(nn.Module):
    def __init__(self, smooth=1):
        super(DiceBCELoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # dice
        intersection = (inputs * targets).sum()
        dice = 1 - (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)

        # bce
        bce = F.binary_cross_entropy(inputs, targets, reduction='mean')

        return bce + dice



# DATASET
class PersonDatasetSubset(Dataset):
    def __init__(self, images_dir, masks_dir, file_pairs, augment=False):
        self.file_pairs = file_pairs
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.augment = augment

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):
        try:
            img_name, mask_name = self.file_pairs[idx]
            img = cv2.imread(os.path.join(self.images_dir, img_name))
            mask = cv2.imread(os.path.join(self.masks_dir, mask_name), cv2.IMREAD_GRAYSCALE)


            if img is None or mask is None:
                raise ValueError(f"Imagine sau masca corupta la indexul {idx}")

            img, mask = cv2.resize(img, (256, 256)), cv2.resize(mask, (256, 256))

            if self.augment:
                # Flip
                if random.random() > 0.5:
                    img, mask = cv2.flip(img, 1), cv2.flip(mask, 1)

                # Rotatie
                if random.random() > 0.5:
                    angle = random.uniform(-10, 10)
                    M = cv2.getRotationMatrix2D((128, 128), angle, 1.0)
                    img = cv2.warpAffine(img, M, (256, 256), flags=cv2.INTER_LINEAR)
                    mask = cv2.warpAffine(mask, M, (256, 256), flags=cv2.INTER_NEAREST)

            img = img.astype(np.float32) / 255.0
            mask = (mask > 127).astype(np.float32)

            return torch.from_numpy(img).permute(2, 0, 1), torch.from_numpy(mask).unsqueeze(0)

        except Exception as e:
            #eroare poze
            new_idx = random.randint(0, len(self.file_pairs) - 1)
            return self.__getitem__(new_idx)



# iou
def compute_iou(pred, target, threshold=0.5):
    pred_bin = (pred > threshold).float()
    intersection = (pred_bin * target).sum()
    union = pred_bin.sum() + target.sum() - intersection
    return (intersection / (union + 1e-6)).item()



# MAIN
if __name__ == "__main__":

    # Config & Split

    images_dir, masks_dir = "../dataset/images", "../dataset/masks"
    max_images = 12188
    batch_size = 32
    num_epochs = 30
    lr = 1e-3

    img_dict = {os.path.splitext(f)[0]: f for f in os.listdir(images_dir) if
                f.lower().endswith(('.jpg', '.jpeg', '.png'))}
    mask_dict = {os.path.splitext(f)[0]: f for f in os.listdir(masks_dir) if
                 f.lower().endswith(('.jpg', '.jpeg', '.png'))}
    common_names = sorted(list(set(img_dict.keys()) & set(mask_dict.keys())))[:max_images]
    all_pairs = [(img_dict[name], mask_dict[name]) for name in common_names]

    random.seed(42)
    random.shuffle(all_pairs)

    train_idx, val_idx = int(0.8 * len(all_pairs)), int(0.9 * len(all_pairs))
    train_files, val_files, test_files = all_pairs[:train_idx], all_pairs[train_idx:val_idx], all_pairs[val_idx:]

    # Loaders
    train_ds = PersonDatasetSubset(images_dir, masks_dir, train_files, augment=True)
    val_ds = PersonDatasetSubset(images_dir, masks_dir, val_files, augment=False)
    test_ds = PersonDatasetSubset(images_dir, masks_dir, test_files, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleUNet().to(device)


    criterion = DiceBCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    # Scheduler: scade LR automat dacă IoU nu mai crește pe validare
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5)

    train_ious, val_ious = [], []
    best_val_iou = 0.0

    print(f"\nIncepe antrenarea pe {device}...")
    print(f"Perechi gasite: {len(all_pairs)}")


    # Training Loop
    for epoch in range(num_epochs):
        model.train()
        train_loss, train_iou = 0, 0

        for i, (imgs, masks) in enumerate(train_loader):
            imgs, masks = imgs.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, masks)  # dicebcce
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
        val_loss, val_iou_epoch = 0, 0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                outputs = model(imgs)
                v_loss = criterion(outputs, masks)
                val_loss += v_loss.item()
                val_iou_epoch += compute_iou(outputs, masks)

        avg_train_iou = (train_iou / len(train_loader)) * 100
        avg_val_iou = (val_iou_epoch / len(val_loader)) * 100

        print("-" * 50)
        print(f"FINAL EPOCA {epoch + 1}")

        train_ious.append(avg_train_iou)
        val_ious.append(avg_val_iou)

        scheduler.step(avg_val_iou / 100)

        print(f"Antrenare - IoU: {avg_train_iou:.2f}%")
        print(f"Validare  - IoU: {avg_val_iou:.2f}% | LR curent: {optimizer.param_groups[0]['lr']:.6f}")

        if (avg_val_iou / 100) > best_val_iou:
            best_val_iou = avg_val_iou / 100
            torch.save(model.state_dict(), "best_model.pth")
            print(f"*** BEST MODEL salvat: {avg_val_iou:.2f}% ***")
        print("-" * 50)

    # final

    print("\n" + "=" * 30)
    print("DATE FINALE pe epoci:")
    print(f"train_ious = {train_ious}")
    print(f"val_ious = {val_ious}")
    print("=" * 30 + "\n")


    # test final
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

    print(f"IOU FINAL PE TEST: {(test_iou_sum / len(test_loader)) * 100:.2f}%")