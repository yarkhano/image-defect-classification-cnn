import os
from PIL import Image, ImageDraw
import random

RAW_DIR = "data/raw"
IMG_SIZE = 64
N_PER_CLASS = 250

os.makedirs(f"{RAW_DIR}/non_defective", exist_ok=True)
os.makedirs(f"{RAW_DIR}/defective", exist_ok=True)

def draw_non_defective(idx):
    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), "white")
    draw = ImageDraw.Draw(img)
    shape = random.choice(["circle", "square"])
    cx, cy, r = IMG_SIZE // 2, IMG_SIZE // 2, 20
    if shape == "circle":
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline="black", fill="gray", width=2)
    else:
        draw.rectangle([cx - r, cy - r, cx + r, cy + r], outline="black", fill="gray", width=2)
    img.save(f"{RAW_DIR}/non_defective/img_{idx}.png")

def draw_defective(idx):
    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), "white")
    draw = ImageDraw.Draw(img)
    shape = random.choice(["circle", "square"])
    # random offset (off-center) + random missing chunk / distortion
    offset_x = random.randint(-15, 15)
    offset_y = random.randint(-15, 15)
    cx, cy, r = IMG_SIZE // 2 + offset_x, IMG_SIZE // 2 + offset_y, random.randint(14, 24)
    if shape == "circle":
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline="black", fill="gray", width=2)
    else:
        draw.rectangle([cx - r, cy - r, cx + r, cy + r], outline="black", fill="gray", width=2)
    # simulate a "missing piece" by painting a white wedge over part of the shape
    draw.pieslice([cx - r, cy - r, cx + r, cy + r], start=0, end=70, fill="white")
    img.save(f"{RAW_DIR}/defective/img_{idx}.png")

for i in range(N_PER_CLASS):
    draw_non_defective(i)
    draw_defective(i)

print(f"Generated {N_PER_CLASS} images per class in {RAW_DIR}/")