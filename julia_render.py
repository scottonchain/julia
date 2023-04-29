import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from matplotlib.colors import hsv_to_rgb

width, height = 1600, 1600
x_range = (-1.98, 1.98)
y_range = (-2.04, 2.04)
max_iter = 300

# Burning Ship fractal
x = np.linspace(x_range[0], x_range[1], width)
y = np.linspace(y_range[0], y_range[1], height)
X, Y = np.meshgrid(x, y)
C = X + 1j * Y
Z = np.zeros_like(C)
ship = np.zeros(C.shape, dtype=int)
mask = np.ones(C.shape, dtype=bool)
for i in range(max_iter):
    Z[mask] = (np.abs(Z[mask].real) + 1j * np.abs(Z[mask].imag)) ** 2 + C[mask]
    mask_new = np.abs(Z) <= 2
    ship[mask & ~mask_new] = i
    mask = mask_new

hsv = np.zeros((height, width, 3), dtype=float)
hsv[..., 0] = (0.2 * ship / max_iter + 0.8) % 1
hsv[..., 1] = 0.9 - 0.7 * (ship / max_iter)
hsv[..., 2] = (ship / max_iter) ** 0.7
rgb = (hsv_to_rgb(hsv) * 255).astype(np.uint8)
img = Image.fromarray(rgb)

# Julia set as transparency mask
c = complex(-0.76, 0.19)
Z2 = X + 1j * Y
julia = np.zeros(Z2.shape, dtype=int)
mask = np.ones(Z2.shape, dtype=bool)
for i in range(max_iter):
    Z2[mask] = Z2[mask] ** 2 + c
    mask_new = np.abs(Z2) <= 2
    julia[mask & ~mask_new] = i
    mask = mask_new
alpha = (julia / max_iter * 255).astype(np.uint8)
img = img.convert('RGBA')
img.putalpha(Image.fromarray(alpha))

# Motion blur effect
img = img.filter(ImageFilter.GaussianBlur(radius=2)).filter(ImageFilter.BoxBlur(3))
img = img.convert('RGB')
img = ImageEnhance.Color(img).enhance(2.1)

output_path = 'julia_output.jpg'
img.save(output_path) 