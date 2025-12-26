import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
import torch
import live_demo as ld
from PIL import Image, ImageTk
from model import SimpleUNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "best_model.pth"
DEFAULT_BG = "background.jpg"
INPUT_SIZE = 256

model = SimpleUNet().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

def load_bg(path):
    bg = cv2.imread(path)
    if bg is None:
        return np.ones((720, 1280, 3), dtype=np.uint8) * 255
    return cv2.resize(bg, (1280, 720))

bg_image = load_bg(DEFAULT_BG)

root = tk.Tk()
root.title("Background Removal")
threshold_var = tk.DoubleVar(value=0.2)
label = tk.Label(root)
label.pack()

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

running = True

def update_frame():
    global bg_image
    if not running:
        return

    ret, frame = cap.read()

    frame = cv2.flip(frame, 1)  # mirror
    current_threshold = threshold_var.get()
    with torch.no_grad():
        mask = model(ld.preprocess_frame(frame))
    mask = ld.postprocess_mask(mask,threshold=current_threshold)

    output = frame * mask + bg_image * (1 - mask)
    output = np.clip(output, 0, 255).astype(np.uint8)

    img = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    imgtk = ImageTk.PhotoImage(image=img)

    label.imgtk = imgtk
    label.config(image=imgtk)

    root.after(10, update_frame)

def change_background():
    global bg_image
    path = filedialog.askopenfilename()
    if path:
        bg_image = load_bg(path)

def stop():
    global running
    running = False
    cap.release()
    root.destroy()

btn_bg = tk.Button(root, text="Schimbă background", command=change_background)
btn_bg.pack(side=tk.LEFT, padx=5)

btn_exit = tk.Button(root, text="Iesire", command=stop)
btn_exit.pack(side=tk.LEFT, padx=5)

lbl_threshold = tk.Label(root, text="Ajustare Threshold:")
lbl_threshold.pack(side=tk.LEFT, padx=5)

scale_threshold = tk.Scale(
    root,
    from_=0.0,
    to=1.0,
    resolution=0.01,
    orient=tk.HORIZONTAL,
    length=300,
    variable=threshold_var
)
scale_threshold.pack(side=tk.LEFT, padx=5)

update_frame()
root.mainloop()
