import numpy as np
from PIL import Image, ImageEnhance
from matplotlib.colors import hsv_to_rgb

width, height = 1600, 1600
x_range = (-1.42, 1.42)
y_range = (-1.53, 1.53)
c = complex(-0.43, 0.56)
max_iter = 1000

x = np.linspace(x_range[0], x_range[1], width)
y = np.linspace(y_range[0], y_range[1], height)
X, Y = np.meshgrid(x, y)
Z = X + 1j * Y

div_iter = np.zeros(Z.shape, dtype=int)
mask = np.ones(Z.shape, dtype=bool)
for i in range(max_iter):
    Z[mask] = Z[mask] ** 2 + c
    mask_new = np.abs(Z) <= 2
    div_iter[mask & ~mask_new] = i
    mask = mask_new

with np.errstate(divide='ignore', invalid='ignore'):
    smooth = div_iter + 1 - np.log(np.log2(np.abs(Z)))
    smooth = np.nan_to_num(smooth)
smooth_norm = (smooth - smooth.min()) / (smooth.max() - smooth.min())

hsv = np.zeros((height, width, 3), dtype=float)
hsv[..., 0] = (0.6 * smooth_norm + 0.3) % 1
hsv[..., 1] = 0.97 - 0.15 * np.abs(np.sin(2 * np.pi * smooth_norm))
hsv[..., 2] = smooth_norm ** 0.2

rgb = (hsv_to_rgb(hsv) * 255).astype(np.uint8)
img = Image.fromarray(rgb)
img = ImageEnhance.Color(img).enhance(2.0)
img = ImageEnhance.Contrast(img).enhance(1.3)

output_path = 'julia_output.jpg'
img.save(output_path) 