import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
from matplotlib.colors import hsv_to_rgb

width, height = 1600, 1600
x_range = (-0.83, 0.83)
y_range = (-0.54, 0.54)
c = complex(-0.77, 0.15)
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
hsv[..., 0] = (0.7 * smooth_norm + 0.2) % 1
hsv[..., 1] = 0.95 - 0.1 * smooth_norm
hsv[..., 2] = smooth_norm ** 0.2

rgb = (hsv_to_rgb(hsv) * 255).astype(np.uint8)
img = Image.fromarray(rgb)

img = ImageOps.solarize(img, threshold=80)

# Circular pixel sort
def circular_pixel_sort(im):
    arr = np.array(im)
    cy, cx = arr.shape[0] // 2, arr.shape[1] // 2
    for r in range(1, min(cy, cx)):
        indices = (np.abs(np.sqrt((np.arange(arr.shape[0])[:, None] - cy) ** 2 + (np.arange(arr.shape[1]) - cx) ** 2) - r) < 1)
        for c in range(3):
            band = arr[..., c][indices]
            band.sort()
            arr[..., c][indices] = band
    return Image.fromarray(arr)

img = circular_pixel_sort(img)
img = img.filter(ImageFilter.EDGE_ENHANCE)

output_path = 'julia_output.jpg'
img.save(output_path) 