import numpy as np
from PIL import Image, ImageDraw, ImageEnhance
from matplotlib.colors import hsv_to_rgb

width, height = 900, 900
x_range = (-2.2, 1.2)
y_range = (-2.0, 1.0)
max_iter = 320

x = np.linspace(x_range[0], x_range[1], width)
y = np.linspace(y_range[0], y_range[1], height)
X, Y = np.meshgrid(x, y)
C = X + 1j * Y
Z = np.zeros_like(C)
burning_ship = np.zeros(C.shape, dtype=int)
mask = np.ones(C.shape, dtype=bool)
for i in range(max_iter):
    Z[mask] = (np.abs(Z[mask].real) + 1j * np.abs(Z[mask].imag)) ** 2 + C[mask]
    mask_new = np.abs(Z) <= 2
    burning_ship[mask & ~mask_new] = i
    mask = mask_new

hsv = np.zeros((height, width, 3), dtype=float)
hsv[..., 0] = (0.1 * burning_ship / max_iter + 0.8) % 1
hsv[..., 1] = 0.7 + 0.3 * (burning_ship / max_iter)
hsv[..., 2] = (burning_ship / max_iter) ** 0.9
rgb = (hsv_to_rgb(hsv) * 255).astype(np.uint8)
img = Image.fromarray(rgb)

# Overlay: grid
step = 0.3
for val in np.arange(x_range[0], x_range[1] + step, step):
    x_pix = int((val - x_range[0]) / (x_range[1] - x_range[0]) * width)
    ImageDraw.Draw(img).line([(x_pix, 0), (x_pix, height)], fill=(255, 200, 200), width=1)
for val in np.arange(y_range[0], y_range[1] + step, step):
    y_pix = int((y_range[1] - val) / (y_range[1] - y_range[0]) * height)
    ImageDraw.Draw(img).line([(0, y_pix), (width, y_pix)], fill=(255, 200, 200), width=1)

img = ImageEnhance.Color(img).enhance(1.3)
img = ImageEnhance.Contrast(img).enhance(1.2)

output_path = 'julia_output.jpg'
img.save(output_path) 