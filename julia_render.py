import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from matplotlib.colors import hsv_to_rgb

width, height = 1600, 1600
x_range = (-1.31, 1.31)
y_range = (-1.31, 1.31)
c = complex(-0.41, 0.65)
max_iter = 340

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
hsv[..., 0] = (0.8 * smooth_norm + 0.2) % 1
hsv[..., 1] = 0.7 + 0.3 * np.abs(np.sin(4 * np.pi * smooth_norm))
hsv[..., 2] = smooth_norm ** 0.5

rgb = (hsv_to_rgb(hsv) * 255).astype(np.uint8)
img = Image.fromarray(rgb)

def wave_warp(im, amp=10, freq=0.1):
    arr = np.array(im)
    for i in range(arr.shape[0]):
        arr[i] = np.roll(arr[i], int(amp * np.sin(freq * i)))
    return Image.fromarray(arr)

img = wave_warp(img, amp=15, freq=0.15)
enhanced = ImageEnhance.Color(img).enhance(2.0)
enhanced = ImageEnhance.Contrast(enhanced).enhance(1.2)

output_path = 'julia_output.jpg'
enhanced.save(output_path) 