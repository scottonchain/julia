import numpy as np
from PIL import Image, ImageEnhance, ImageDraw
from matplotlib.colors import hsv_to_rgb

width, height = 1600, 1600
x_range = (-1.58, 1.58)
y_range = (-1.58, 1.58)
c = complex(-0.7, 0.26)
max_iter = 500

# Generate grid of complex points
x = np.linspace(x_range[0], x_range[1], width)
y = np.linspace(y_range[0], y_range[1], height)
X, Y = np.meshgrid(x, y)
Z = X + 1j * Y

# Initialize iteration counts and mask
div_iter = np.zeros(Z.shape, dtype=int)
mask = np.ones(Z.shape, dtype=bool)

# Iterate and record divergence with standard tolerance
for i in range(max_iter):
    Z[mask] = Z[mask] ** 2 + c
    mask_new = np.abs(Z) <= 2
    div_iter[mask & ~mask_new] = i
    mask = mask_new

# Smooth coloring
with np.errstate(divide='ignore', invalid='ignore'):
    smooth = div_iter + 1 - np.log(np.log2(np.abs(Z)))
    smooth = np.nan_to_num(smooth)
smooth_norm = (smooth - smooth.min()) / (smooth.max() - smooth.min())

# Build HSV image with a bright color scheme
hsv = np.zeros((height, width, 3), dtype=float)
hsv[..., 0] = (0.7 * smooth_norm + 0.2) % 1
hsv[..., 1] = 0.95 - 0.1 * np.abs(np.sin(2 * np.pi * smooth_norm))
hsv[..., 2] = smooth_norm ** 0.2

# Convert to RGB
rgb = (hsv_to_rgb(hsv) * 255).astype(np.uint8)
img = Image.fromarray(rgb)

# Create a second image with different tolerance (escape radius = 1.5)
Z2 = X + 1j * Y
div_iter2 = np.zeros(Z2.shape, dtype=int)
mask2 = np.ones(Z2.shape, dtype=bool)

for i in range(max_iter):
    Z2[mask2] = Z2[mask2] ** 2 + c
    mask_new2 = np.abs(Z2) <= 1.5  # Different tolerance
    div_iter2[mask2 & ~mask_new2] = i
    mask2 = mask_new2

with np.errstate(divide='ignore', invalid='ignore'):
    smooth2 = div_iter2 + 1 - np.log(np.log2(np.abs(Z2)))
    smooth2 = np.nan_to_num(smooth2)
smooth_norm2 = (smooth2 - smooth2.min()) / (smooth2.max() - smooth2.min())

hsv2 = np.zeros((height, width, 3), dtype=float)
hsv2[..., 0] = (0.8 * smooth_norm2 + 0.1) % 1
hsv2[..., 1] = 0.98 - 0.2 * np.abs(np.sin(2 * np.pi * smooth_norm2))
hsv2[..., 2] = smooth_norm2 ** 0.2

rgb2 = (hsv_to_rgb(hsv2) * 255).astype(np.uint8)
img2 = Image.fromarray(rgb2)

# Create a third image with very high tolerance (escape radius = 3.0)
Z3 = X + 1j * Y
div_iter3 = np.zeros(Z3.shape, dtype=int)
mask3 = np.ones(Z3.shape, dtype=bool)

for i in range(max_iter):
    Z3[mask3] = Z3[mask3] ** 2 + c
    mask_new3 = np.abs(Z3) <= 3.0  # High tolerance
    div_iter3[mask3 & ~mask_new3] = i
    mask3 = mask_new3

with np.errstate(divide='ignore', invalid='ignore'):
    smooth3 = div_iter3 + 1 - np.log(np.log2(np.abs(Z3)))
    smooth3 = np.nan_to_num(smooth3)
smooth_norm3 = (smooth3 - smooth3.min()) / (smooth3.max() - smooth3.min())

hsv3 = np.zeros((height, width, 3), dtype=float)
hsv3[..., 0] = (0.6 * smooth_norm3 + 0.3) % 1
hsv3[..., 1] = 0.97 - 0.15 * np.abs(np.sin(2 * np.pi * smooth_norm3))
hsv3[..., 2] = smooth_norm3 ** 0.2

rgb3 = (hsv_to_rgb(hsv3) * 255).astype(np.uint8)
img3 = Image.fromarray(rgb3)

# Combine the three images side by side
combined_width = width * 3
combined_img = Image.new('RGB', (combined_width, height))

# Paste the three images
combined_img.paste(img, (0, 0))
combined_img.paste(img2, (width, 0))
combined_img.paste(img3, (width * 2, 0))

# Add labels
draw = ImageDraw.Draw(combined_img)
draw.text((10, 10), "Tolerance = 2.0", fill=(255, 255, 255))
draw.text((width + 10, 10), "Tolerance = 1.5", fill=(255, 255, 255))
draw.text((width * 2 + 10, 10), "Tolerance = 3.0", fill=(255, 255, 255))

# Enhance the combined image
enhanced = ImageEnhance.Color(combined_img).enhance(1.5)
enhanced = ImageEnhance.Contrast(enhanced).enhance(1.2)

# Save output
output_path = 'julia_output.jpg'
enhanced.save(output_path) 