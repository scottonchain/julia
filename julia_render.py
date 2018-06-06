import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw
from matplotlib.colors import hsv_to_rgb

width, height = 1600, 1600
x_range = (-1.6, 1.6)
y_range = (-1.6, 1.6)
c = complex(0.39, 0.38)
max_iter = 360

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
hsv[..., 0] = (0.4 * smooth_norm + 0.4) % 1
hsv[..., 1] = 0.9 + 0.1 * np.abs(np.sin(2 * np.pi * smooth_norm))
hsv[..., 2] = smooth_norm ** 0.3

rgb = (hsv_to_rgb(hsv) * 255).astype(np.uint8)
img = Image.fromarray(rgb)

img = img.filter(ImageFilter.EMBOSS)

# Circular swirl mask overlay
def swirl_mask(im):
    arr = np.array(im)
    cy, cx = arr.shape[0] // 2, arr.shape[1] // 2
    Y, X = np.ogrid[:arr.shape[0], :arr.shape[1]]
    r = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)
    mask = (np.sin(r / 20) > 0)
    arr[mask] = arr[mask] // 2
    return Image.fromarray(arr)

img = swirl_mask(img)
img = ImageEnhance.Contrast(img).enhance(1.5)

output_path = 'julia_output.jpg'
img.save(output_path) 