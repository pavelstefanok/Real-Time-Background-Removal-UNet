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
# train_ious = [60.1774269073713, 70.72381564828217, 75.34394246632937, 78.05941052124149, 79.65839855006483, 80.76548742466285, 81.5908510567712, 82.59531144235956, 83.16165134555004, 83.67694293866393, 83.93694729101463, 84.35741158782459, 84.31574426713537, 85.18895211766979, 85.22382222238134, 85.59654378500142, 85.74604059829086, 85.9816529711739, 86.26612637863784, 86.17428418065681, 86.56778740101173, 86.51293537655815, 87.95429046036767, 88.07899793640513, 88.3935896490441, 88.31844955194191, 88.4175841730149, 88.62754643940535, 88.78941107968815, 88.80963436892775]
# val_ious = [64.70257632243329, 74.83678505970882, 75.56668260158636, 80.233590113811, 79.90122345777658, 83.65263113608728, 83.07946905111655, 83.35207456197494, 84.73418813485367, 84.37804304636441, 85.51211494665878, 84.64548465533134, 86.21784616739322, 86.261835312232, 84.99164489599374, 86.10088382011804, 85.85283144926413, 86.53709017313443, 85.48159400622049, 85.40128438900678, 85.77191355900887, 84.98991788961948, 88.52312366167703, 88.30868999163309, 87.82103382624112, 88.32302979933911, 88.73754235414359, 88.44286631315182, 87.94636833362091, 88.0585606281574]
# ==============================
#
# TEST FINAL
# IOU FINAL PE TEST: 87.79%

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