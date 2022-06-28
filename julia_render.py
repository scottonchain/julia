import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
from matplotlib.colors import hsv_to_rgb

width, height = 1600, 1600
x_range = (-1.56, 1.56)
y_range = (-1.64, 1.64)
c = complex(0.33, 0.34)
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
hsv[..., 0] = (0.7 * smooth_norm + 0.2) % 1
hsv[..., 1] = 0.9 + 0.1 * np.abs(np.sin(2 * np.pi * smooth_norm))
hsv[..., 2] = smooth_norm ** 0.4

rgb = (hsv_to_rgb(hsv) * 255).astype(np.uint8)
img = Image.fromarray(rgb)

img = img.filter(ImageFilter.EMBOSS)

# Checkerboard mask overlay
def checkerboard(im, size=40):
    arr = np.array(im)
    for i in range(0, arr.shape[0], size):
        for j in range(0, arr.shape[1], size):
            if (i // size + j // size) % 2 == 0:
                arr[i:i+size, j:j+size] = arr[i:i+size, j:j+size] // 2
    return Image.fromarray(arr)

img = checkerboard(img, size=50)
img = ImageEnhance.Color(img).enhance(1.7)

output_path = 'julia_output.jpg'
img.save(output_path) 