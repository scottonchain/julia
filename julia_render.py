import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
from matplotlib.colors import hsv_to_rgb

width, height = 1600, 1600
x_range = (-0.71, 0.71)
y_range = (-0.64, 0.64)
c = complex(0.41, -0.18)
max_iter = 380

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

img = img.filter(ImageFilter.FIND_EDGES)

# Horizontal flip every 100 pixels
def flip_blocks(im, block=100):
    arr = np.array(im)
    for i in range(0, arr.shape[0], block*2):
        arr[i:i+block] = arr[i:i+block][::-1]
    return Image.fromarray(arr)

img = flip_blocks(img, block=100)
img = ImageEnhance.Contrast(img).enhance(1.8)

output_path = 'julia_output.jpg'
img.save(output_path) 