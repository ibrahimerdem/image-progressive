import os
import numpy as np
from PIL import Image

DIRS = [
    "outputs/diffusion/generated/",
    "outputs/diffusion/target/",
]

STD_THRESH = 5 


def content_bbox(strip):

    col_std = strip.std(axis=(0, 2))
    row_std = strip.std(axis=(1, 2))
    content_cols = np.where(col_std > STD_THRESH)[0]
    content_rows = np.where(row_std > STD_THRESH)[0]

    left = int(content_cols[0]) if len(content_cols) else 0
    right = int(content_cols[-1]) + 1 if len(content_cols) else strip.shape[1]
    top = int(content_rows[0]) if len(content_rows) else 0
    bot = int(content_rows[-1]) + 1 if len(content_rows) else strip.shape[0]
    return left, right, top, bot


def crop_dir(src_dir):
    dst_dir = src_dir + "_cropped"
    os.makedirs(dst_dir, exist_ok=True)
    files = sorted(f for f in os.listdir(src_dir) if f.lower().endswith(".jpg"))
    for fname in files:
        img = Image.open(os.path.join(src_dir, fname)).convert("RGB")
        W, H = img.size
        v_top    = int(H * 0.60)
        v_bottom = int(H * 0.98)
        strip = np.array(img)[v_top:v_bottom, :]
        left, right, s_top, s_bot = content_bbox(strip)
        cropped = Image.fromarray(strip[s_top:s_bot, left:right])
        cropped.save(os.path.join(dst_dir, fname))
    print(f"{src_dir}  ->  {dst_dir}  ({len(files)} images)")


for d in DIRS:
    if os.path.isdir(d):
        crop_dir(d)
    else:
        print(f"Skipped (not found): {d}")