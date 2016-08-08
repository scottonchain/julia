import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from matplotlib.colors import hsv_to_rgb

width, height = 1600, 1600
x_range = (-0.72, 0.72)
y_range = (-0.52, 0.52)
c = complex(0.36, -0.42)
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

# Duotone palette
def duotone(im, color1=(30, 30, 120), color2=(220, 220, 60)):
    arr = np.array(im).astype(np.float32) / 255.0
    mask = arr[..., 0] > 0.5
    arr[mask] = np.array(color1) / 255.0
    arr[~mask] = np.array(color2) / 255.0
    return Image.fromarray((arr * 255).astype(np.uint8))

img = duotone(img)
img = img.filter(ImageFilter.EDGE_ENHANCE_MORE)

output_path = 'julia_output.jpg'
img.save(output_path) 