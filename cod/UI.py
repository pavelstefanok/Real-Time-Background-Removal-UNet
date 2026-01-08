import customtkinter as ctk
from tkinter import filedialog
import cv2
import numpy as np
import torch
import live_demo as ld
from PIL import Image
from model import SimpleUNet

# ==============================
# DATE FINALE pe epoci:
# train_ious = [45.22255552992833, 54.269283464739225, 58.8979263106167, 61.57491105907368, 63.667190272950855, 65.18275635080127, 66.07940466898792, 66.74247855187636, 67.47301486156533, 68.22255100012804, 68.6729279329192, 69.29335816655616]
# val_ious = [48.57500333801594, 60.02911677937103, 62.523790356380495, 65.11008285619076, 68.22327784463471, 66.70383318187365, 68.95044948540482, 66.11354004323871, 70.15746890329847, 65.25213835286159, 65.56652505802953, 70.1068457435159]
# ==============================
#
# TEST FINAL
# IOU FINAL PE TEST: 68.65%

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

ctk.set_appearance_mode("dark")
root = ctk.CTk()
root.title("Background Removal")

threshold_var = ctk.DoubleVar(value=0.2)
view_mode = ctk.StringVar(value="Final")  # Modul de vizualizare default

label = ctk.CTkLabel(root, text="")
label.pack(pady=10)

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

running = True


def update_frame():
    global bg_image
    if not running: return

    ret, frame = cap.read()
    if not ret: return

    frame = cv2.flip(frame, 1)
    current_threshold = threshold_var.get()
    current_mode = view_mode.get()

    with torch.no_grad():
        mask = model(ld.preprocess_frame(frame))
    mask = ld.postprocess_mask(mask, threshold=current_threshold)

    # Logica de Switch pentru afisare
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


# --- CONTROALE JOS ---
controls_frame = ctk.CTkFrame(root)
controls_frame.pack(side="bottom", fill="x", padx=10, pady=10)

btn_bg = ctk.CTkButton(controls_frame, text="Schimbă background", command=change_background)
btn_bg.pack(side="left", padx=5)

# Switch-ul de vizualizare (Original / Masca / Final)
mode_switch = ctk.CTkSegmentedButton(controls_frame, values=["Original", "Mască", "Final"],
                                     variable=view_mode)
mode_switch.pack(side="left", padx=10)

lbl_threshold = ctk.CTkLabel(controls_frame, text="Threshold:")
lbl_threshold.pack(side="left", padx=5)

scale_threshold = ctk.CTkSlider(controls_frame, from_=0.0, to=1.0, variable=threshold_var, width=200)
scale_threshold.pack(side="left", padx=5)

btn_exit = ctk.CTkButton(controls_frame, text="Iesire", fg_color="red", command=stop)
btn_exit.pack(side="right", padx=5)

update_frame()
root.mainloop()