import numpy as np
from PIL import Image, ImageDraw, ImageEnhance
from matplotlib.colors import hsv_to_rgb

width, height = 900, 900
x_range = (-2.0, 2.0)
y_range = (-2.0, 2.0)
max_iter = 340
p = 0.5667 - 0.5j

x = np.linspace(x_range[0], x_range[1], width)
y = np.linspace(y_range[0], y_range[1], height)
X, Y = np.meshgrid(x, y)
C = X + 1j * Y
Z = np.zeros_like(C)
Zold = np.zeros_like(C)
phoenix = np.zeros(C.shape, dtype=int)
mask = np.ones(C.shape, dtype=bool)
for i in range(max_iter):
    Z[mask], Zold[mask] = Z[mask] ** 2 + C[mask] + p * Zold[mask], Z[mask]
    mask_new = np.abs(Z) <= 2
    phoenix[mask & ~mask_new] = i
    mask = mask_new

hsv = np.zeros((height, width, 3), dtype=float)
hsv[..., 0] = (0.5 * phoenix / max_iter + 0.2) % 1
hsv[..., 1] = 0.7 + 0.3 * (phoenix / max_iter)
hsv[..., 2] = (phoenix / max_iter) ** 0.85
rgb = (hsv_to_rgb(hsv) * 255).astype(np.uint8)
img = Image.fromarray(rgb)

# Overlay: grid
step = 0.4
for val in np.arange(x_range[0], x_range[1] + step, step):
    x_pix = int((val - x_range[0]) / (x_range[1] - x_range[0]) * width)
    ImageDraw.Draw(img).line([(x_pix, 0), (x_pix, height)], fill=(200, 255, 220), width=1)
for val in np.arange(y_range[0], y_range[1] + step, step):
    y_pix = int((y_range[1] - val) / (y_range[1] - y_range[0]) * height)
    ImageDraw.Draw(img).line([(0, y_pix), (width, y_pix)], fill=(200, 255, 220), width=1)

img = ImageEnhance.Color(img).enhance(1.5)
img = ImageEnhance.Contrast(img).enhance(1.2)

output_path = 'julia_output.jpg'
img.save(output_path) 