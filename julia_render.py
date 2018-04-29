import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw
from matplotlib.colors import hsv_to_rgb

width, height = 1600, 1600
x_range = (-1.54, 1.54)
y_range = (-1.46, 1.46)
c = complex(-0.81, 0.13)
max_iter = 400

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

# Deep blue palette
hsv = np.zeros((height, width, 3), dtype=float)
hsv[..., 0] = 0.6 + 0.1 * smooth_norm
hsv[..., 1] = 0.7 + 0.3 * smooth_norm
hsv[..., 2] = smooth_norm ** 0.8

rgb = (hsv_to_rgb(hsv) * 255).astype(np.uint8)
img = Image.fromarray(rgb)

img = img.filter(ImageFilter.GaussianBlur(radius=10))

# Radial gradient overlay
def radial_gradient(im):
    arr = np.array(im).astype(np.float32)
    cy, cx = arr.shape[0] // 2, arr.shape[1] // 2
    Y, X = np.ogrid[:arr.shape[0], :arr.shape[1]]
    r = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)
    mask = r / r.max()
    for c in range(3):
        arr[..., c] = arr[..., c] * (1 - 0.5 * mask)
    return Image.fromarray(arr.astype(np.uint8))

img = radial_gradient(img)
img = ImageEnhance.Brightness(img).enhance(1.1)

output_path = 'julia_output.jpg'
img.save(output_path) 