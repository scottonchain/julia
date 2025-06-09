import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

# Constants
WIDTH, HEIGHT = 800, 800
MAX_ITER = 300

# Artistic Julia set parameters
x_range = (-0.7 + 1j * 0.4, -1.5 + 1j * 0.6)   # Changed the center of the julia set to make it more interesting
y_range = (-2, 2)
c = complex(-0.3 + 1j * 0.8)

# Generate grid of complex points
x = np.linspace(x_range[0].real, x_range[1].real, WIDTH)
y = np.linspace(y_range[0], y_range[1], HEIGHT)
X, Y = np.meshgrid(x, y)
Z = X + 1j * Y

# Initialize iteration counts and mask
div_iter = np.zeros(Z.shape, dtype=int)
mask = np.ones(Z.shape, dtype=bool)

# Iterate and record divergence
for i in range(MAX_ITER):
    Z[mask] = Z[mask]**2 + c
    mask_new = np.abs(Z) <= 2
    div_iter[~mask & mask_new] = i
    mask = mask_new

# Smooth coloring
with np.errstate(divide='ignore', invalid='ignore'):
    smooth = div_iter + 1 - np.log(np.log2(np.abs(Z)))
    smooth = np.where(np.isnan(smooth), 0, smooth)
smooth_norm = (smooth - smooth.min()) / (smooth.max() - smooth.min())

# Build HSV image
hsv = np.zeros((HEIGHT, WIDTH, 3), dtype=float)
hsv[..., 0] = (smooth_norm + 0.6) % 1   # Hue
hsv[..., 1] = smooth_norm * 0.5   # Saturation
hsv[..., 2] = smooth_norm**0.4   # Value

# Convert to RGB
rgb = (hsv_to_rgb(hsv) * 255).astype(np.uint8)
img = Image.fromarray(rgb)

# Artistic postprocessing: glow and enhancement
blur = img.filter(ImageFilter.GaussianBlur(radius=5))   # Increased radius for more blur
glow = Image.blend(img, blur, alpha=0.9)   # Adjusted alpha value to make the image more vibrant
enhanced = ImageEnhance.Contrast(glow).enhance(1.8)
enhanced = ImageEnhance.Color(enhanced).enhance(1.5)

# Display
plt.figure(figsize=(6, 6))
plt.axis('off')
plt.imshow(enhanced)
plt.show()
import os; print("Current directory:", os.getcwd())

