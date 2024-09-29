import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

# Artistic Julia set parameters
width, height = 800, 800
x_range = (-1.5 + 0.2j, -0.8 + 0.156j)  # Change the center of the julia set
y_range = (-1.5, 1.5)
c = complex(-0.8, 0.156)  # Tweak this for different shapes
max_iter = 300

# Generate grid of complex points
x = np.linspace(x_range[0].real, x_range[1].real, width)
y = np.linspace(y_range[0], y_range[1], height)
X, Y = np.meshgrid(x + 1j * (y - y_range[0]), y - y_range[0])
Z = X + 1j * Y

# Initialize iteration counts and mask
div_iter = np.zeros(Z.shape, dtype=int)
mask = np.ones(Z.shape, dtype=bool)

# Iterate and record divergence
for i in range(max_iter):
    Z[mask] = Z[mask]**2 + c
    mask_new = np.abs(Z) <= 2
    div_iter[mask & ~mask_new] = i
    mask = mask_new

# Smooth coloring
with np.errstate(divide='ignore', invalid='ignore'):
    smooth = div_iter + 1 - np.log(np.log2(np.abs(Z)))
    smooth = np.nan_to_num(smooth)
smooth_norm = (smooth - smooth.min()) / (smooth.max() - smooth.min())

# Build HSV image
hsv = np.zeros((height, width, 3), dtype=float)
hsv[..., 0] = (smooth_norm + 0.6) % 1  # Hue
hsv[..., 1] = 0.8 + 0.2 * smooth_norm  # Saturation
hsv[..., 2] = smooth_norm ** 0.3  # Value

# Convert to RGB
rgb = (hsv_to_rgb(hsv) * 255).astype(np.uint8)
img = Image.fromarray(rgb)

# Artistic postprocessing: glow and enhancement
blur = img.filter(ImageFilter.GaussianBlur(radius=4))  # Change the radius of the blur filter
glow = Image.blend(img, blur, alpha=0.5)  # Change the alpha value for a more subtle glow effect
enhanced = ImageEnhance.Contrast(glow).enhance(1.3)
enhanced = ImageEnhance.Color(enhanced).enhance(1.2)

# Display
plt.figure(figsize=(6, 6))
plt.axis('off')
plt.imshow(enhanced)
plt.show()

