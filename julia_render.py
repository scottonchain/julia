import numpy as np
from PIL import Image, ImageDraw, ImageEnhance
from matplotlib.colors import hsv_to_rgb

width, height = 1600, 1600
x_range = (-2.02, 2.02)
y_range = (-1.97, 1.97)
c = complex(-0.84, 0.16)
max_iter = 300

# Julia set
x = np.linspace(x_range[0], x_range[1], width)
y = np.linspace(y_range[0], y_range[1], height)
X, Y = np.meshgrid(x, y)
Z = X + 1j * Y
julia = np.zeros(Z.shape, dtype=int)
mask = np.ones(Z.shape, dtype=bool)
for i in range(max_iter):
    Z[mask] = Z[mask] ** 2 + c
    mask_new = np.abs(Z) <= 2
    julia[mask & ~mask_new] = i
    mask = mask_new

# Bright rainbow palette
hsv = np.zeros((height, width, 3), dtype=float)
hsv[..., 0] = (julia / max_iter + 0.3) % 1
hsv[..., 1] = 0.9 + 0.1 * (julia / max_iter)
hsv[..., 2] = (julia / max_iter) ** 0.5
rgb = (hsv_to_rgb(hsv) * 255).astype(np.uint8)
img = Image.fromarray(rgb)

# Zoomed Mandelbrot overlay
mandel = np.zeros(Z.shape, dtype=int)
C = (X/2) + 1j * (Y/2)
Z2 = np.zeros_like(C)
mask = np.ones(C.shape, dtype=bool)
for i in range(max_iter):
    Z2[mask] = Z2[mask] ** 2 + C[mask]
    mask_new = np.abs(Z2) <= 2
    mandel[mask & ~mask_new] = i
    mask = mask_new
mandel_img = (mandel / max_iter * 255).astype(np.uint8)
mandel_img = Image.fromarray(np.stack([mandel_img]*3, axis=-1)).convert('RGBA')
mandel_img.putalpha(80)
img = img.convert('RGBA')
img = Image.alpha_composite(img, mandel_img)
img = img.convert('RGB')

# Draw random geometric shapes
draw = ImageDraw.Draw(img)
for _ in range(30):
    shape = np.random.choice(['ellipse', 'rectangle', 'line'])
    xy = [np.random.randint(0, width), np.random.randint(0, height), np.random.randint(0, width), np.random.randint(0, height)]
    color = tuple(np.random.randint(100, 255, 3))
    if shape == 'ellipse':
        draw.ellipse(xy, outline=color, width=2)
    elif shape == 'rectangle':
        draw.rectangle(xy, outline=color, width=2)
    else:
        draw.line(xy, fill=color, width=2)

img = ImageEnhance.Color(img).enhance(2.0)
img = ImageEnhance.Contrast(img).enhance(1.2)

output_path = 'julia_output.jpg'
img.save(output_path) 