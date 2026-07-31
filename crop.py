import os
import glob
import argparse
import numpy as np
import cv2

TOP_SEARCH_FRACTION = 0.30
TOP_CROP_START = 0.1
TOP_CROP_HEIGHT = 0.2

SIDE_MARGIN_RATIO = 0.27
BACKGROUND_BRIGHTNESS_THRESHOLD = 235

MIN_RECT_AREA = 400

SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png")


def find_trouser_mask_and_bbox(img_bgr):
    h, w = img_bgr.shape[:2]

    if BACKGROUND_BRIGHTNESS_THRESHOLD is None:
        return np.full((h, w), 255, dtype=np.uint8), (0, 0, w, h)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    raw_mask = (gray < BACKGROUND_BRIGHTNESS_THRESHOLD).astype(np.uint8) * 255

    kernel = np.ones((5, 5), np.uint8)
    raw_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel)
    raw_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(raw_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.full((h, w), 255, dtype=np.uint8), (0, 0, w, h)

    largest = max(contours, key=cv2.contourArea)
    x, y, bw, bh = cv2.boundingRect(largest)
    if bw * bh < 0.05 * w * h:
        return np.full((h, w), 255, dtype=np.uint8), (0, 0, w, h)

    filled_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(filled_mask, [largest], -1, 255, thickness=-1)

    return filled_mask, (x, y, bw, bh)


def maximal_foreground_rectangle(sub_mask):
    rows, cols = sub_mask.shape
    if rows == 0 or cols == 0:
        return None

    height = [0] * cols
    best_area = 0
    best_rect = None
    binary = (sub_mask > 0)

    for i in range(rows):
        for j in range(cols):
            if binary[i, j]:
                height[j] += 1
            else:
                height[j] = 0

        stack = []
        j = 0
        while j <= cols:
            h_ = height[j] if j < cols else 0
            if not stack or h_ >= height[stack[-1]]:
                stack.append(j)
                j += 1
            else:
                top_idx = stack.pop()
                left = (stack[-1] + 1) if stack else 0
                right = j - 1
                width = right - left + 1
                area = height[top_idx] * width
                if area > best_area:
                    best_area = area
                    top = i - height[top_idx] + 1
                    best_rect = (top, left, i, right)

    if best_rect is None or best_area < MIN_RECT_AREA:
        return None
    return best_rect


def inner_crop(img_bgr, mask, bbox, y0, y1):
    x, y, w, h = bbox

    # Narrow the crop horizontally
    side = int(w * SIDE_MARGIN_RATIO)
    x0 = x + side
    x1 = x + w - side

    y0c = max(y, y0)
    y1c = min(y + h, y1)

    if y1c <= y0c:
        y0c, y1c = y, y + h

    sub_mask = mask[y0c:y1c, x0:x1]
    rect = maximal_foreground_rectangle(sub_mask)

    if rect is None:
        return img_bgr[y0c:y1c, x0:x1]

    top, left, bottom, right = rect

    abs_y0 = y0c + top
    abs_y1 = y0c + bottom + 1
    abs_x0 = x0 + left
    abs_x1 = x0 + right + 1

    return img_bgr[abs_y0:abs_y1, abs_x0:abs_x1]


def get_crop_bounds(img_bgr, bbox):
    x, y, w, h = bbox

    top_y0 = y + int(h * TOP_CROP_START)
    top_y1 = y + int(h * (TOP_CROP_START + TOP_CROP_HEIGHT))

    top_y0 = max(y, top_y0)
    top_y1 = min(y + h, top_y1)

    return top_y0, top_y1


def process_image(path, output_dir):
    img = cv2.imread(path)
    if img is None:
        print(f"  [skip] could not read: {path}")
        return

    mask, bbox = find_trouser_mask_and_bbox(img)

    top_y0, top_y1 = get_crop_bounds(img, bbox)

    top_crop = inner_crop(img, mask, bbox, top_y0, top_y1)

    base = os.path.splitext(os.path.basename(path))[0]
    ext = ".jpg"

    top_path = os.path.join(output_dir, f"{base}{ext}")

    if top_crop.size:
        cv2.imwrite(top_path, top_crop)
        print(f"  [ok] {base}: top={top_crop.shape[:2]}")
    else:
        print(f"  [warn] empty crop for {base}")


def main():
    data_dir = "data"
    default_input_dir = os.path.join(data_dir, "target")
    default_output_dir = os.path.join(data_dir, "cropped")

    parser = argparse.ArgumentParser(description="Extract background-free regions")
    parser.add_argument("--input", default=default_input_dir, help="source")
    parser.add_argument("--output", default=default_output_dir, help="target")
    parser.add_argument("--num", default=None, type=int, help="number of images to process")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    files = []
    for ext in SUPPORTED_EXTENSIONS:
        files.extend(glob.glob(os.path.join(args.input, f"*{ext}")))
        files.extend(glob.glob(os.path.join(args.input, f"*{ext.upper()}")))
    files = sorted(set(files))

    if not files:
        print(f"No images found in {args.input}")
        return

    print(f"Found {len(files)} image(s). Processing...")
    for i, path in enumerate(files):
        if args.num is not None and i >= args.num:
            break
        process_image(path, args.output)

    print("Done.")


if __name__ == "__main__":
    main()