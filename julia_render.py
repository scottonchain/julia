import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

# Parameters (these will be programmatically changed by the main script)
width, height = 1600, 1600
x_range = (-1.62, 1.62)
y_range = (-1.71, 1.71)
c = complex(-0.44, 0.58)
max_iter = 350

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

# Build HSV image with a bright color scheme
hsv = np.zeros((height, width, 3), dtype=float)
hsv[..., 0] = (0.6 * smooth_norm + 0.1) % 1  # Bright hue shift
hsv[..., 1] = 0.9 + 0.1 * np.sin(2 * np.pi * smooth_norm)  # High saturation
hsv[..., 2] = smooth_norm ** 0.3  # Bright value

# Convert to RGB
rgb = (hsv_to_rgb(hsv) * 255).astype(np.uint8)
img = Image.fromarray(rgb)

# Kaleidoscope effect (4-way mirror)
def kaleidoscope(im):
    arr = np.array(im)
    arr = np.concatenate([arr, arr[:, ::-1]], axis=1)
    arr = np.concatenate([arr, arr[::-1, :]], axis=0)
    return Image.fromarray(arr)

img = kaleidoscope(img)

# Artistic postprocessing: glow, contrast, and edge enhancement
blur = img.filter(ImageFilter.GaussianBlur(radius=3))
glow = Image.blend(img, blur, alpha=0.4)
enhanced = ImageEnhance.Contrast(glow).enhance(1.8)
enhanced = ImageEnhance.Color(enhanced).enhance(1.5)
enhanced = enhanced.filter(ImageFilter.EDGE_ENHANCE_MORE)

# Save output
output_path = 'julia_output.jpg'
enhanced.save(output_path)

# Optionally display
# plt.figure(figsize=(8, 8))
# plt.axis('off')
# plt.imshow(enhanced)
# plt.show() 