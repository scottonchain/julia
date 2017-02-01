import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

# Parameters (these will be programmatically changed by the main script)
width, height = 1600, 1600
x_range = (-0.74, 0.74)
y_range = (-0.69, 0.69)
c = complex(-0.73, -0.42)
max_iter = 500

# Generate grid of complex points
x = np.linspace(x_range[0], x_range[1], width)
y = np.linspace(y_range[0], y_range[1], height)
X, Y = np.meshgrid(x, y)
Z = X + 1j * Y

# Initialize iteration counts and mask
div_iter = np.zeros(Z.shape, dtype=int)
mask = np.ones(Z.shape, dtype=bool)

# Iterate and record divergence
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

# Build HSV image with a unique color scheme
hsv = np.zeros((height, width, 3), dtype=float)
hsv[..., 0] = (0.7 * smooth_norm + 0.2) % 1  # Brighter hue
hsv[..., 1] = 0.95 - 0.2 * np.abs(np.cos(3 * np.pi * smooth_norm))  # High saturation
hsv[..., 2] = smooth_norm ** 0.4  # Bright value

# Convert to RGB
rgb = (hsv_to_rgb(hsv) * 255).astype(np.uint8)
img = Image.fromarray(rgb)

# Mirrored effect
def mirror(im):
    return ImageOps.mirror(im)

img = mirror(img)

# Solarize effect
img = ImageOps.solarize(img, threshold=128)

# Artistic postprocessing: blur, color, and emboss
blur = img.filter(ImageFilter.GaussianBlur(radius=2))
enhanced = ImageEnhance.Color(blur).enhance(1.8)
enhanced = enhanced.filter(ImageFilter.EMBOSS)

# Save output
output_path = 'julia_output.jpg'
enhanced.save(output_path)

# Optionally display
# plt.figure(figsize=(10, 8))
# plt.axis('off')
# plt.imshow(enhanced)
# plt.show() 