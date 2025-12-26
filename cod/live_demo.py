import cv2
import numpy as np
import torch
from model import SimpleUNet


# configuratii

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model_path = "best_model.pth"
bg_path = "background.jpg"

model = SimpleUNet().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print("Model loaded.")


# background

bg = cv2.imread(bg_path)
if bg is None:
    bg = np.ones((720, 1280, 3), dtype=np.uint8) * 255
else:
    bg = cv2.resize(bg, (1280, 720), interpolation=cv2.INTER_LINEAR)

#pre post procesare
def preprocess_frame(frame, input_size=256):
    frame_resized = cv2.resize(frame, (input_size, input_size))
    frame_norm = frame_resized / 255.0
    frame_tensor = torch.from_numpy(frame_norm).permute(2, 0, 1).unsqueeze(0).float()
    return frame_tensor.to(device)


def postprocess_mask(mask, original_size=(1280, 720), threshold=0.2):

        mask_np = mask.squeeze().cpu().detach().numpy()
        mask_bin = (mask_np > threshold).astype(np.uint8)
        mask_up = cv2.resize(mask_bin, original_size, interpolation=cv2.INTER_NEAREST)

        # mare
        # num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_up)
        # if num_labels > 1:
        #     largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        #     mask_up = (labels == largest_label).astype(np.uint8)

        # eroz dilat
        kernel = np.ones((7, 7), np.uint8)
        mask_up = cv2.morphologyEx(mask_up, cv2.MORPH_OPEN, kernel)
        mask_up = cv2.morphologyEx(mask_up, cv2.MORPH_CLOSE, kernel)

        # 3 can
        mask_up = mask_up[..., np.newaxis]
        mask_up = np.repeat(mask_up, 3, axis=2)
        return mask_up

