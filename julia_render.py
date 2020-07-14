import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
from matplotlib.colors import hsv_to_rgb

width, height = 1600, 1600
x_range = (-1.74, 1.74)
y_range = (-1.74, 1.74)
c = complex(-0.74, -0.35)
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

# Pastel palette
hsv = np.zeros((height, width, 3), dtype=float)
hsv[..., 0] = (0.7 * smooth_norm + 0.2) % 1
hsv[..., 1] = 0.4 + 0.3 * np.abs(np.sin(2 * np.pi * smooth_norm))
hsv[..., 2] = smooth_norm ** 0.5

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

# Vertical split mirror
def vertical_split_mirror(im):
    arr = np.array(im)
    mid = arr.shape[1] // 2
    arr[:, mid:] = arr[:, :mid][:, ::-1]
    return Image.fromarray(arr)

img = vertical_split_mirror(img)
img = ImageEnhance.Color(img).enhance(1.5)

output_path = 'julia_output.jpg'
img.save(output_path) 