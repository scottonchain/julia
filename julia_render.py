import numpy as np
from PIL import Image, ImageEnhance
from matplotlib.colors import hsv_to_rgb

width, height = 1600, 1600
x_range = (-1.54, 1.54)
y_range = (-1.54, 1.54)
max_iter = 300

# Multibrot (power 3)
x = np.linspace(x_range[0], x_range[1], width)
y = np.linspace(y_range[0], y_range[1], height)
X, Y = np.meshgrid(x, y)
C = X + 1j * Y
Z = np.zeros_like(C)
multi = np.zeros(C.shape, dtype=int)
mask = np.ones(C.shape, dtype=bool)
for i in range(max_iter):
    Z[mask] = Z[mask] ** 3 + C[mask]
    mask_new = np.abs(Z) <= 2
    multi[mask & ~mask_new] = i
    mask = mask_new

hsv = np.zeros((height, width, 3), dtype=float)
hsv[..., 0] = (0.6 * multi / max_iter + 0.2) % 1
hsv[..., 1] = 0.5 + 0.5 * (multi / max_iter)
hsv[..., 2] = (multi / max_iter) ** 0.8
rgb = (hsv_to_rgb(hsv) * 255).astype(np.uint8)
img = Image.fromarray(rgb)

# Julia set as mask
c = complex(-0.82, 0.17)
Z2 = X + 1j * Y
julia = np.zeros(Z2.shape, dtype=int)
mask = np.ones(Z2.shape, dtype=bool)
for i in range(max_iter):
    Z2[mask] = Z2[mask] ** 2 + c
    mask_new = np.abs(Z2) <= 2
    julia[mask & ~mask_new] = i
    mask = mask_new
alpha = (julia / max_iter * 255).astype(np.uint8)
img = img.convert('RGBA')
img.putalpha(Image.fromarray(alpha))

# Channel mixing
arr = np.array(img)
arr[..., 0], arr[..., 1], arr[..., 2] = arr[..., 2], arr[..., 0], arr[..., 1]
img = Image.fromarray(arr, 'RGBA')

# Transparency gradient
grad = np.linspace(0, 255, height).astype(np.uint8)
grad = np.tile(grad[:, None], (1, width))
arr = np.array(img)
arr[..., 3] = (arr[..., 3].astype(np.float32) * grad / 255).astype(np.uint8)
img = Image.fromarray(arr, 'RGBA').convert('RGB')

img = ImageEnhance.Color(img).enhance(1.5)

output_path = 'julia_output.jpg'
img.save(output_path) 