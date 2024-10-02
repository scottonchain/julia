import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw, ImageOps
from matplotlib.colors import hsv_to_rgb

width, height = 1600, 1600
x_range = (-1.35, 1.35)
y_range = (-1.41, 1.41)
c = complex(-0.72, -0.35)
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
hsv[..., 0] = (0.08 + 0.12 * smooth_norm) % 1
hsv[..., 1] = 0.8 - 0.5 * smooth_norm
hsv[..., 2] = smooth_norm ** 0.7

rgb = (hsv_to_rgb(hsv) * 255).astype(np.uint8)
img = Image.fromarray(rgb)

img = ImageOps.solarize(img, threshold=120)

# Horizontal banding overlay
def add_bands(im, band_height=30):
    arr = np.array(im)
    for y in range(0, arr.shape[0], band_height*2):
        arr[y:y+band_height] = arr[y:y+band_height] // 2
    return Image.fromarray(arr)

img = add_bands(img, band_height=40)
img = ImageEnhance.Color(img).enhance(1.3)

output_path = 'julia_output.jpg'
img.save(output_path) 