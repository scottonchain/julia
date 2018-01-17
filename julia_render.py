import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
from matplotlib.colors import hsv_to_rgb

width, height = 1600, 1600
x_range = (-2.08, 2.08)
y_range = (-2.08, 2.08)
c = complex(-0.68, -0.37)
max_iter = 350

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
hsv[..., 0] = (0.7 * smooth_norm + 0.2) % 1
hsv[..., 1] = 0.95 - 0.1 * smooth_norm
hsv[..., 2] = smooth_norm ** 0.2

rgb = (hsv_to_rgb(hsv) * 255).astype(np.uint8)
img = Image.fromarray(rgb)

# Heavy pixelation
def pixelate(im, block=20):
    arr = np.array(im)
    for i in range(0, arr.shape[0], block):
        for j in range(0, arr.shape[1], block):
            arr[i:i+block, j:j+block] = arr[i, j]
    return Image.fromarray(arr)

img = pixelate(img, block=30)
img = ImageOps.flip(img)
img = ImageEnhance.Color(img).enhance(2.0)

output_path = 'julia_output.jpg'
img.save(output_path) 