import numpy as np
from PIL import Image, ImageDraw, ImageEnhance
from matplotlib.colors import hsv_to_rgb

width, height = 1600, 1600
x_range = (-2.02, 2.02)
y_range = (-1.97, 1.97)
max_iter = 300

# Tricorn fractal
x = np.linspace(x_range[0], x_range[1], width)
y = np.linspace(y_range[0], y_range[1], height)
X, Y = np.meshgrid(x, y)
C = X + 1j * Y
Z = np.zeros_like(C)
tricorn = np.zeros(C.shape, dtype=int)
mask = np.ones(C.shape, dtype=bool)
for i in range(max_iter):
    Z[mask] = np.conj(Z[mask]) ** 2 + C[mask]
    mask_new = np.abs(Z) <= 2
    tricorn[mask & ~mask_new] = i
    mask = mask_new

hsv = np.zeros((height, width, 3), dtype=float)
hsv[..., 0] = (0.2 * tricorn / max_iter + 0.8) % 1
hsv[..., 1] = 0.9 + 0.1 * (tricorn / max_iter)
hsv[..., 2] = (tricorn / max_iter) ** 0.5
rgb = (hsv_to_rgb(hsv) * 255).astype(np.uint8)
img = Image.fromarray(rgb)

# Julia set overlay
c = complex(0.24, -0.0)
Z2 = X + 1j * Y
julia = np.zeros(Z2.shape, dtype=int)
mask = np.ones(Z2.shape, dtype=bool)
for i in range(max_iter):
    Z2[mask] = Z2[mask] ** 2 + c
    mask_new = np.abs(Z2) <= 2
    julia[mask & ~mask_new] = i
    mask = mask_new
julia_img = (julia / max_iter * 255).astype(np.uint8)
julia_img = Image.fromarray(np.stack([julia_img]*3, axis=-1)).convert('RGBA')
julia_img.putalpha(80)
img = img.convert('RGBA')
img = Image.alpha_composite(img, julia_img)
img = img.convert('RGB')

# Color cycling effect
arr = np.array(img)
arr = np.roll(arr, shift=30, axis=2)
img = Image.fromarray(arr)

# Geometric grid overlay
draw = ImageDraw.Draw(img)
for x in range(0, width, 60):
    draw.line((x, 0, x, height), fill=(255,255,255,60), width=1)
for y in range(0, height, 60):
    draw.line((0, y, width, y), fill=(255,255,255,60), width=1)

img = ImageEnhance.Color(img).enhance(1.7)

output_path = 'julia_output.jpg'
img.save(output_path) 