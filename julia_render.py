import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from matplotlib.colors import hsv_to_rgb

width, height = 1600, 1600
x_range = (-0.66, 0.66)
y_range = (-0.66, 0.66)
c = complex(-0.74, 0.25)
max_iter = 500

x = np.linspace(x_range[0], x_range[1], width)
y = np.linspace(y_range[0], y_range[1], height)
X, Y = np.meshgrid(x, y)
Z = X + 1j * Y

iteration = np.zeros(Z.shape, dtype=int)
mask = np.ones(Z.shape, dtype=bool)

for i in range(max_iter):
    Z[mask] = Z[mask] ** 2 + c
    mask_new = np.abs(Z) <= 2
    iteration[mask & ~mask_new] = i
    mask = mask_new

# Smooth coloring
with np.errstate(divide='ignore', invalid='ignore'):
    smooth = iteration + 1 - np.log(np.log2(np.abs(Z)))
    smooth = np.nan_to_num(smooth)
smooth_norm = (smooth - smooth.min()) / (smooth.max() - smooth.min())

# Bright palette
hsv = np.zeros((height, width, 3), dtype=float)
hsv[..., 0] = (0.7 * smooth_norm + 0.2) % 1
hsv[..., 1] = 0.95 - 0.1 * np.abs(np.sin(2 * np.pi * smooth_norm))
hsv[..., 2] = smooth_norm ** 0.2

rgb = (hsv_to_rgb(hsv) * 255).astype(np.uint8)

# Create PIL image and apply enhancements
img = Image.fromarray(rgb)

# Apply PIL filters and enhancements
img = img.filter(ImageFilter.EDGE_ENHANCE_MORE)
img = img.filter(ImageFilter.SMOOTH_MORE)

# Enhance colors and contrast
enhancer = ImageEnhance.Color(img)
img = enhancer.enhance(1.5)

enhancer = ImageEnhance.Contrast(img)
img = enhancer.enhance(1.3)

enhancer = ImageEnhance.Brightness(img)
img = enhancer.enhance(1.1)

# Save output
output_path = 'julia_output.jpg'
img.save(output_path, quality=95) 