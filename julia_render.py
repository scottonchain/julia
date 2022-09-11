import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

# Parameters (these will be programmatically changed by the main script)
width, height = 1600, 1600
x_range = (-0.76, 0.76)
y_range = (-0.81, 0.81)
c = complex(0.4, 0.36)
max_iter = 320

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
hsv[..., 0] = (0.6 * smooth_norm + 0.3) % 1  # Brighter hue
hsv[..., 1] = 0.95 - 0.1 * np.cos(4 * np.pi * smooth_norm)  # High saturation
hsv[..., 2] = smooth_norm ** 0.3  # Bright value

# Convert to RGB
rgb = (hsv_to_rgb(hsv) * 255).astype(np.uint8)
img = Image.fromarray(rgb)

# Circular ripple postprocessing effect
def circular_ripple(im, freq=12, amp=8):
    arr = np.array(im)
    cy, cx = arr.shape[0] // 2, arr.shape[1] // 2
    Y, X = np.ogrid[:arr.shape[0], :arr.shape[1]]
    r = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)
    ripple = amp * np.sin(2 * np.pi * r / freq)
    newY = np.clip((Y + ripple).astype(int), 0, arr.shape[0] - 1)
    newX = np.clip((X + ripple).astype(int), 0, arr.shape[1] - 1)
    rippled = arr[newY, newX]
    return Image.fromarray(rippled)

img = circular_ripple(img, freq=18, amp=10)

# Artistic postprocessing: blur, color, and detail
blur = img.filter(ImageFilter.GaussianBlur(radius=2))
enhanced = ImageEnhance.Color(blur).enhance(1.7)
enhanced = ImageEnhance.Contrast(enhanced).enhance(1.3)
enhanced = enhanced.filter(ImageFilter.DETAIL)

# Save output
output_path = 'julia_output.jpg'
enhanced.save(output_path) 