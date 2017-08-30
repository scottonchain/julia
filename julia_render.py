import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageOps
from matplotlib.colors import hsv_to_rgb

width, height = 1600, 1600
x_range = (-1.97, 1.97)
y_range = (-1.97, 1.97)
c = complex(-0.71, 0.22)
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

# Mandelbrot mask
mandel = np.zeros(Z.shape, dtype=int)
C = X + 1j * Y
Z2 = np.zeros_like(C)
mask = np.ones(C.shape, dtype=bool)
for i in range(max_iter):
    Z2[mask] = Z2[mask] ** 2 + C[mask]
    mask_new = np.abs(Z2) <= 2
    mandel[mask & ~mask_new] = i
    mask = mask_new

# Vibrant color map
hsv = np.zeros((height, width, 3), dtype=float)
hsv[..., 0] = (0.9 * julia / max_iter + 0.1) % 1
hsv[..., 1] = 0.8 + 0.2 * (mandel / max_iter)
hsv[..., 2] = (julia / max_iter) ** 0.5
rgb = (hsv_to_rgb(hsv) * 255).astype(np.uint8)
img = Image.fromarray(rgb)

# Hand-drawn scribble effect
scribble = Image.new('RGBA', img.size, (0,0,0,0))
draw = ImageDraw.Draw(scribble)
for _ in range(200):
    x0, y0 = np.random.randint(0, width), np.random.randint(0, height)
    x1, y1 = x0 + np.random.randint(-30, 30), y0 + np.random.randint(-30, 30)
    color = tuple(np.random.randint(100, 255, 3)) + (np.random.randint(40, 100),)
    draw.line((x0, y0, x1, y1), fill=color, width=np.random.randint(1, 4))
img = img.convert('RGBA')
img = Image.alpha_composite(img, scribble)
img = img.convert('RGB')

img = ImageEnhance.Color(img).enhance(2.2)
img = ImageEnhance.Contrast(img).enhance(1.3)

output_path = 'julia_output.jpg'
img.save(output_path) 