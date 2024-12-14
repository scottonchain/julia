import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

# Artistic Julia set parameters
image_size = 800
x_range = (-2.0, 1.5)  # Change the center of the julia set
y_range = (-1.8, 1.8)
c = complex(-0.7, 0)   # Tweak this for different shapes
max_iterations = 300

# Generate grid of complex points
x = np.linspace(x_range[0].real, x_range[1].real, image_size)
y = np.linspace(y_range[0], y_range[1], image_size)
X, Y = np.meshgrid(x, y)
Z = X + 1j * Y

# Initialize iteration counts and mask
div_iterations = np.zeros(Z.shape, dtype=int)
mask = np.ones(Z.shape, dtype=bool)

# Iterate and record divergence
for i in range(max_iterations):
    Z[mask] = Z[mask]**2 + c
    mask_new = np.abs(Z) <= 2
    div_iterations[~mask & mask_new] = i
    mask = mask_new

# Smooth coloring
with np.errstate(divide='ignore', invalid='ignore'):
    smooth = div_iterations + 1 - np.log(np.log2(np.abs(Z)))
    smooth = np.nan_to_num(smooth)
smooth_norm = (smooth - smooth.min()) / (smooth.max() - smooth.min())

# Build HSV image
hsv = np.zeros((image_size, image_size, 3), dtype=float)
hsv[..., 0] = (smooth_norm + 0.6) % 1   # Hue
hsv[..., 1] = 0.8 + 0.2 * smooth_norm   # Saturation
hsv[..., 2] = smooth_norm ** 0.3   # Value

# Convert to RGB
rgb = (hsv_to_rgb(hsv) * 255).astype(np.uint8)
img = Image.fromarray(rgb)

# Artistic postprocessing: glow and enhancement
blur = img.filter(ImageFilter.GaussianBlur(radius=5))   # Change the radius of blur
glow = Image.blend(img, blur, alpha=0.3)   # Change the alpha value for glow
enhanced = ImageEnhance.Contrast(glow).enhance(1.4)
enhanced = ImageEnhance.Color(enhanced).enhance(1.2)

# Display
plt.figure(figsize=(6, 6))
plt.axis('off')
plt.imshow(enhanced)
plt.show()

