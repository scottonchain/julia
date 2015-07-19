import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

width, height = 1600, 1600
x_range = (-1.58, 1.58)
y_range = (-1.64, 1.64)
c = complex(-0.81, 0.13)
max_iter = 370

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
hsv[..., 0] = (0.5 * smooth_norm + 0.3) % 1
hsv[..., 1] = 0.95 - 0.3 * np.abs(np.sin(2 * np.pi * smooth_norm))
hsv[..., 2] = smooth_norm ** 0.4

rgb = (hsv_to_rgb(hsv) * 255).astype(np.uint8)
img = Image.fromarray(rgb)

def glass_distort(im, scale=6):
    arr = np.array(im)
    dx = (np.random.rand(*arr.shape[:2]) - 0.5) * scale
    dy = (np.random.rand(*arr.shape[:2]) - 0.5) * scale
    Y, X = np.meshgrid(np.arange(arr.shape[0]), np.arange(arr.shape[1]), indexing='ij')
    Xn = np.clip((X + dx).astype(int), 0, arr.shape[1] - 1)
    Yn = np.clip((Y + dy).astype(int), 0, arr.shape[0] - 1)
    distorted = arr[Yn, Xn]
    return Image.fromarray(distorted)

img = glass_distort(img, scale=8)

blur = img.filter(ImageFilter.GaussianBlur(radius=1))
enhanced = ImageEnhance.Color(blur).enhance(2.2)
enhanced = ImageEnhance.Contrast(enhanced).enhance(1.5)

output_path = 'julia_output.jpg'
enhanced.save(output_path) 