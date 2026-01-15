import customtkinter as ctk
from tkinter import filedialog
import cv2
import numpy as np
import torch
import os
import sys
import live_demo as ld
from PIL import Image
from model import SimpleUNet


def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = resource_path("best_model.pth")
DEFAULT_BG = resource_path("background.jpg")

# Incarcare model
model = SimpleUNet().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
model.eval()


def load_bg(path):
    bg = cv2.imread(path)
    if bg is None:
        return np.ones((720, 1280, 3), dtype=np.uint8) * 255
    return cv2.resize(bg, (1280, 720))


bg_image = load_bg(DEFAULT_BG)

ctk.set_appearance_mode("dark")
root = ctk.CTk()
root.title(f"AI Background Remover - {DEVICE}")

threshold_var = ctk.DoubleVar(value=0.3)
view_mode = ctk.StringVar(value="Final")
largest_comp_var = ctk.BooleanVar(value=False)

label = ctk.CTkLabel(root, text="")
label.pack(pady=10)

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

running = True


def update_frame():
    global bg_image
    if not running: return

    try:
        ret, frame = cap.read()
        if not ret:
            root.after(10, update_frame)
            return

        frame = cv2.flip(frame, 1)
        current_threshold = threshold_var.get()
        current_mode = view_mode.get()

        # Procesare AI (doar daca nu suntem pe Original)
        if current_mode != "Original":
            with torch.no_grad():
                input_tensor = ld.preprocess_frame(frame).to(DEVICE)
                mask_out = model(input_tensor)

            mask = ld.postprocess_mask(
                mask_out,
                threshold=current_threshold,
                use_largest=largest_comp_var.get()
            )

        # Logica de vizualizare
        if current_mode == "Final":
            output = frame * mask + bg_image * (1 - mask)
        elif current_mode == "Mască":
            output = (mask * 255).astype(np.uint8)
        else:  # Original
            output = frame

        output = np.clip(output, 0, 255).astype(np.uint8)
        img = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
        ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(1280, 720))

        label.configure(image=ctk_img)
        label.image = ctk_img

    except Exception as e:
        print(f"Eroare Frame: {e}")

    finally:
        root.after(10, update_frame)


def change_background():
    global bg_image
    path = filedialog.askopenfilename()
    if path: bg_image = load_bg(path)


def stop():
    global running
    running = False
    cap.release()
    root.destroy()


# --- CONTROALE ---
controls_frame = ctk.CTkFrame(root)
controls_frame.pack(side="bottom", fill="x", padx=10, pady=10)

btn_bg = ctk.CTkButton(controls_frame, text="Schimbă background", command=change_background)
btn_bg.pack(side="left", padx=5)

mode_switch = ctk.CTkSegmentedButton(controls_frame, values=["Original", "Mască", "Final"], variable=view_mode)
mode_switch.pack(side="left", padx=10)

# Switch-ul nou pentru Largest Component
sw_largest = ctk.CTkSwitch(controls_frame, text="Clean Mask", variable=largest_comp_var)
sw_largest.pack(side="left", padx=10)

lbl_threshold = ctk.CTkLabel(controls_frame, text="Threshold:")
lbl_threshold.pack(side="left", padx=5)

scale_threshold = ctk.CTkSlider(controls_frame, from_=0.0, to=1.0, variable=threshold_var, width=150)
scale_threshold.pack(side="left", padx=5)

btn_exit = ctk.CTkButton(controls_frame, text="Iesire", fg_color="red", command=stop)
btn_exit.pack(side="right", padx=5)

update_frame()
root.mainloop()