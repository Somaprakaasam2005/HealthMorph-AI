import cv2
import numpy as np


def generate_heatmap(bgr_image: np.ndarray, bbox, out_path):
    h, w = bgr_image.shape[:2]

    if bbox is None:
        cx, cy, bw, bh = w // 2, h // 2, w // 3, h // 3
    else:
        x, y, bw, bh = bbox
        cx = int(x + bw / 2)
        cy = int(y + bh / 2)

    xv, yv = np.meshgrid(np.arange(w), np.arange(h))
    sigma_x = max(20, bw / 2)
    sigma_y = max(20, bh / 2)
    gauss = np.exp(-(((xv - cx) ** 2) / (2 * sigma_x ** 2) + ((yv - cy) ** 2) / (2 * sigma_y ** 2)))

    heat = (gauss * 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(bgr_image, 0.7, heat_color, 0.6, 0)
    cv2.imwrite(out_path, overlay)
