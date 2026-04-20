"""
Defect area detection for fabric images using edge detection and texture analysis.
No paired data required — works on a single image.

Pipeline:
  0. Foreground segmentation → garment mask (exclude white background)
  1. Background subtraction (large-scale Gaussian) → local brightness anomaly
  2. Structural Canny edges (high sigma, high thresh) → tears / holes
  3. Local variance deviation from fabric median → texture regularity breaks
  4. Majority-vote combination within garment interior (edges excluded)
  5. Morphological cleanup + region labelling
  6. Overlay and save results
"""

import os
import numpy as np
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import morphology, color
from skimage.feature import canny

# ── Config ───────────────────────────────────────────────────────────────────
IMAGE_PATH   = "data/target/tip1-recete25-repl2.jpg"
OUTPUT_DIR   = "outputs/defect_detection"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Foreground mask (garment vs. white background)
FG_THRESH      = 0.88    # pixels brighter than this in L channel → background
FG_ERODE_R     = 50      # erode foreground mask to exclude seam/edge regions from detection

# Background field estimation
BG_SIGMA = 40

# Brightness anomaly thresholds (relative to local background)
BG_HIGH_THRESH = 0.16
BG_LOW_THRESH  = 0.14

# Structural Canny
CANNY_SIGMA        = 4.0
CANNY_LOW_THRESH   = 0.90
CANNY_HIGH_THRESH  = 0.97
EDGE_DILATE_R      = 6

# Local variance (z-score)
VAR_WINDOW         = 41
VAR_ZSCORE_THRESH  = 3.0

# Voting
VOTE_THRESH = 0.60

# Morphological cleanup
MORPH_CLOSE_R  = 20
MORPH_OPEN_R   = 8
MIN_REGION_PX  = 400

# ── Load image ────────────────────────────────────────────────────────────────
print(f"Loading image: {IMAGE_PATH}")
img_pil  = Image.open(IMAGE_PATH).convert("RGB")
img_rgb  = np.array(img_pil, dtype=np.float32) / 255.0
img_gray = color.rgb2gray(img_rgb)
H, W     = img_gray.shape
print(f"  Size: {W}x{H}")

# ── Step 0: Foreground garment mask ──────────────────────────────────────────
print("Step 0: Foreground segmentation …")
# White background has very high luminance; garment is mid-tone
fg_mask = img_gray < FG_THRESH

# Fill internal holes (pockets, labels, etc.)
fg_mask = ndimage.binary_fill_holes(fg_mask)

# Remove noise
fg_mask = morphology.remove_small_objects(fg_mask, min_size=5000)

# Erode to stay well inside garment, away from boundary seam edges
selem_erode = morphology.disk(FG_ERODE_R)
fg_interior = morphology.binary_erosion(fg_mask, selem_erode)
print(f"  Garment pixels: {fg_mask.sum()}  Interior pixels: {fg_interior.sum()}")

# ── Step 1: Background subtraction → brightness anomaly ───────────────────────
print("Step 1: Background subtraction …")
bg       = ndimage.gaussian_filter(img_gray, sigma=BG_SIGMA)
residual = img_gray - bg

bright_mask = residual >  BG_HIGH_THRESH
dark_mask   = residual < -BG_LOW_THRESH
bg_anomaly  = (bright_mask | dark_mask).astype(np.float32)

# ── Step 2: Structural Canny edges ───────────────────────────────────────────
print("Step 2: Structural edge detection …")
edges = canny(img_gray, sigma=CANNY_SIGMA,
              low_threshold=CANNY_LOW_THRESH,
              high_threshold=CANNY_HIGH_THRESH,
              use_quantiles=True)

selem_edge   = morphology.disk(EDGE_DILATE_R)
edges_filled = morphology.binary_dilation(edges, selem_edge)
edge_score   = edges_filled.astype(np.float32)

# ── Step 3: Local variance anomaly ───────────────────────────────────────────
print("Step 3: Local variance anomaly …")

def local_variance(gray, window):
    mean  = ndimage.uniform_filter(gray,      size=window)
    mean2 = ndimage.uniform_filter(gray ** 2, size=window)
    return np.clip(mean2 - mean ** 2, 0, None)

lvar = local_variance(img_gray, VAR_WINDOW)

# Compute stats only over the garment interior to avoid background bias
interior_vals = lvar[fg_interior]
med_var = np.median(interior_vals) if interior_vals.size > 0 else np.median(lvar)
std_var = np.std(interior_vals)    if interior_vals.size > 0 else np.std(lvar)
var_anomaly = (np.abs(lvar - med_var) > VAR_ZSCORE_THRESH * std_var).astype(np.float32)

# ── Step 4: Majority-vote + constrain to garment interior ─────────────────────
print("Step 4: Combining signals (majority vote, within interior) …")
combined = (bg_anomaly + edge_score + var_anomaly) / 3.0
raw_mask = (combined >= VOTE_THRESH) & fg_interior   # only keep interior hits

# ── Step 5: Morphological cleanup ────────────────────────────────────────────
print("Step 5: Morphological cleanup …")
selem_close = morphology.disk(MORPH_CLOSE_R)
mask_closed = morphology.binary_closing(raw_mask, selem_close)
# Re-apply interior constraint after closing
mask_closed = mask_closed & fg_interior

selem_open  = morphology.disk(MORPH_OPEN_R)
mask_opened = morphology.binary_opening(mask_closed, selem_open)

mask_clean = morphology.remove_small_objects(mask_opened, min_size=MIN_REGION_PX)

labeled, num_features = ndimage.label(mask_clean)
print(f"  Detected {num_features} defect region(s)")

regions = ndimage.find_objects(labeled)
bboxes  = []
for i, sl in enumerate(regions):
    if sl is None:
        continue
    y0, y1 = sl[0].start, sl[0].stop
    x0, x1 = sl[1].start, sl[1].stop
    area    = int(mask_clean[sl].sum())
    bboxes.append((i + 1, y0, x0, y1, x1, area))
    print(f"    Region {i+1}: y={y0}–{y1}  x={x0}–{x1}  area={area}px")

# ── Step 6: Visualize & save ──────────────────────────────────────────────────
print("Saving outputs …")

def norm_to_uint8(arr):
    a = arr - arr.min()
    denom = a.max()
    if denom < 1e-8:
        return np.zeros_like(arr, dtype=np.uint8)
    return (a / denom * 255).astype(np.uint8)

img_pil.save(os.path.join(OUTPUT_DIR, "00_original.jpg"))

fg_uint8 = (fg_interior.astype(np.uint8)) * 255
Image.fromarray(fg_uint8).save(os.path.join(OUTPUT_DIR, "01_garment_interior_mask.png"))

residual_disp = np.clip(residual * 0.5 + 0.5, 0, 1)
Image.fromarray((residual_disp * 255).astype(np.uint8)).save(
    os.path.join(OUTPUT_DIR, "02_bg_residual.png"))

edge_img = Image.fromarray(edges.astype(np.uint8) * 255)
edge_img.save(os.path.join(OUTPUT_DIR, "03_structural_canny.png"))

lvar_uint8 = norm_to_uint8(lvar)
Image.fromarray(lvar_uint8).save(os.path.join(OUTPUT_DIR, "04_local_variance.png"))

mask_uint8 = (mask_clean.astype(np.uint8)) * 255
Image.fromarray(mask_uint8).save(os.path.join(OUTPUT_DIR, "05_defect_mask.png"))

# Red overlay with bounding boxes
overlay_arr = (img_rgb * 255).astype(np.uint8).copy()
overlay_arr[mask_clean, 0] = np.clip(overlay_arr[mask_clean, 0].astype(int) + 130, 0, 255)
overlay_arr[mask_clean, 1] = (overlay_arr[mask_clean, 1] * 0.3).astype(np.uint8)
overlay_arr[mask_clean, 2] = (overlay_arr[mask_clean, 2] * 0.3).astype(np.uint8)
overlay_pil = Image.fromarray(overlay_arr)

draw = ImageDraw.Draw(overlay_pil)
for (rid, y0, x0, y1, x1, area) in bboxes:
    draw.rectangle([x0, y0, x1, y1], outline=(255, 255, 0), width=4)
    label_y = max(0, y0 - 18)
    draw.text((x0 + 4, label_y), f"R{rid} {area}px", fill=(255, 255, 0))

overlay_pil.save(os.path.join(OUTPUT_DIR, "06_defect_overlay.jpg"))

# Summary panel
DISP_H = 700
scale  = DISP_H / H
dW, dH = max(1, int(W * scale)), max(1, int(H * scale))

def pil_resize(src, size):
    if isinstance(src, np.ndarray):
        src = Image.fromarray(src)
    return src.resize(size, Image.LANCZOS)

panels = [
    (pil_resize(img_pil, (dW, dH)), "Original"),
    (pil_resize(Image.fromarray(fg_uint8).convert("RGB"), (dW, dH)), "Garment Interior"),
    (pil_resize(Image.fromarray((residual_disp*255).astype(np.uint8)).convert("RGB"), (dW, dH)), "BG Residual"),
    (pil_resize(Image.fromarray(lvar_uint8).convert("RGB"), (dW, dH)), "Local Variance"),
    (pil_resize(Image.fromarray(mask_uint8).convert("RGB"), (dW, dH)), "Defect Mask"),
    (pil_resize(overlay_pil, (dW, dH)), "Overlay + BBoxes"),
]

fig, axes = plt.subplots(1, 6, figsize=(24, 9))
for ax, (panel, title) in zip(axes, panels):
    ax.imshow(panel)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.axis("off")

total_defect_px = int(mask_clean.sum())
defect_ratio    = total_defect_px / (H * W) * 100
fig.suptitle(
    f"Defect Detection — {os.path.basename(IMAGE_PATH)}\n"
    f"Regions: {num_features}   Defect area: {defect_ratio:.2f}%   "
    f"FG_erode={FG_ERODE_R}  BG_σ={BG_SIGMA}  Canny_σ={CANNY_SIGMA}  VarWin={VAR_WINDOW}",
    fontsize=11)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "07_summary_panel.png"), dpi=120, bbox_inches="tight")
plt.close(fig)

# ── Report ────────────────────────────────────────────────────────────────────
print("\n── Detection summary ──────────────────────────────────")
print(f"  Image            : {IMAGE_PATH}  ({W}x{H})")
print(f"  Defect regions   : {num_features}")
print(f"  Defect pixels    : {total_defect_px} / {H*W}  ({defect_ratio:.2f}%)")
print(f"  Outputs saved to : {OUTPUT_DIR}/")
print("────────────────────────────────────────────────────────")
