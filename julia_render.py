import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from matplotlib.colors import hsv_to_rgb

width, height = 1600, 1600
x_range = (-1.41, 1.41)
y_range = (-1.41, 1.41)
c = complex(-0.84, 0.2)
max_iter = 400

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
hsv[..., 0] = (0.3 * smooth_norm + 0.7) % 1
hsv[..., 1] = 0.95 - 0.3 * smooth_norm
hsv[..., 2] = smooth_norm ** 0.7

rgb = (hsv_to_rgb(hsv) * 255).astype(np.uint8)
img = Image.fromarray(rgb)

img = img.filter(ImageFilter.FIND_EDGES)
enhanced = ImageEnhance.Contrast(img).enhance(2.5)
enhanced = ImageEnhance.Color(enhanced).enhance(1.8)

output_path = 'julia_output.jpg'
enhanced.save(output_path) 