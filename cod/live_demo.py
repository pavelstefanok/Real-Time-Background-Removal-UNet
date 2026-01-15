import cv2
import numpy as np
import torch
import os
import sys
from model import SimpleUNet

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess_frame(frame, input_size=256):
    frame_resized = cv2.resize(frame, (input_size, input_size))
    frame_norm = frame_resized / 255.0
    frame_tensor = torch.from_numpy(frame_norm).permute(2, 0, 1).unsqueeze(0).float()
    return frame_tensor.to(device)

def postprocess_mask(mask, original_size=(1280, 720), threshold=0.2, use_largest=False):
    mask_np = mask.squeeze().cpu().detach().numpy()
    mask_bin = (mask_np > threshold).astype(np.uint8)
    mask_up = cv2.resize(mask_bin, original_size, interpolation=cv2.INTER_NEAREST)

    # LARGEST COMPONENT
    if use_largest:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_up)
        if num_labels > 1:

            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            mask_up = (labels == largest_label).astype(np.uint8)


    # Eroziune/Dilatare
    kernel = np.ones((7, 7), np.uint8)
    mask_up = cv2.morphologyEx(mask_up, cv2.MORPH_OPEN, kernel)
    mask_up = cv2.morphologyEx(mask_up, cv2.MORPH_CLOSE, kernel)
    # Blur
    mask_up = cv2.GaussianBlur(mask_up, (7, 7), 0)
    # Conversie la 3 canale
    mask_up = mask_up[..., np.newaxis]
    mask_up = np.repeat(mask_up, 3, axis=2)
    return mask_up

if __name__ == "__main__":
    print("Device detectat:", device)