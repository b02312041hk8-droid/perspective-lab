import colorsys

import cv2
import numpy as np
from PIL import Image


def extract_features(image_path, resize_width=400):
    img = Image.open(image_path).convert("RGB")

    original_w, original_h = img.size
    aspect_ratio = original_w / original_h

    if original_w > resize_width:
        new_h = int(original_h * resize_width / original_w)
        img = img.resize((resize_width, new_h))

    arr = np.array(img).astype(np.float32)

    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]

    brightness = (r + g + b) / 3
    mean_brightness = float(np.mean(brightness))
    contrast = float(np.std(brightness))

    flat = arr.reshape(-1, 3) / 255.0
    hsv = np.array([
        colorsys.rgb_to_hsv(px[0], px[1], px[2])
        for px in flat
    ])

    hue = hsv[:, 0] * 360
    saturation = hsv[:, 1] * 100
    value = hsv[:, 2] * 100

    white_mask = (r > 230) & (g > 230) & (b > 230)
    black_mask = (r < 35) & (g < 35) & (b < 35)

    quantized = (arr // 32).astype(np.uint8)
    color_count = len(np.unique(quantized.reshape(-1, 3), axis=0))

    gray = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    edge_ratio = float(np.mean(edges > 0))

    green_mask = (
        (hue >= 70) & (hue <= 170) &
        (saturation >= 15) &
        (value >= 20)
    )

    sky_like_mask = (
        (
            (hue >= 180) & (hue <= 240) &
            (saturation >= 8) &
            (value >= 55)
        )
        |
        (
            (saturation <= 12) &
            (value >= 80)
        )
    )

    warm_mask = (
        ((hue <= 60) | (hue >= 330)) &
        (saturation >= 12) &
        (value >= 20)
    )

    cool_mask = (
        (hue >= 180) & (hue <= 280) &
        (saturation >= 12) &
        (value >= 20)
    )

    return {
        "mean_r": round(float(np.mean(r)), 2),
        "mean_g": round(float(np.mean(g)), 2),
        "mean_b": round(float(np.mean(b)), 2),
        "mean_hue": round(float(np.mean(hue)), 2),
        "mean_brightness": round(mean_brightness, 2),
        "mean_saturation": round(float(np.mean(saturation)), 2),
        "white_ratio": round(float(np.mean(white_mask)), 4),
        "black_ratio": round(float(np.mean(black_mask)), 4),
        "color_count": int(color_count),
        "contrast": round(contrast, 2),
        "edge_ratio": round(edge_ratio, 4),
        "aspect_ratio": round(aspect_ratio, 3),
        "green_ratio": round(float(np.mean(green_mask)), 4),
        "sky_like_ratio": round(float(np.mean(sky_like_mask)), 4),
        "warm_ratio": round(float(np.mean(warm_mask)), 4),
        "cool_ratio": round(float(np.mean(cool_mask)), 4),
    }
