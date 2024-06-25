__doc__   = "Julia Set Visualization"
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import matplotlib.pyplot as plt

# Constants
WIDTH, HEIGHT  = 800, 800
X_RANGE  = (0.5 + 1j * 0, 0.6 + 1j * 0)    # Change the center of the julia set
Y_RANGE  = (-1.5, 1.5)
C  = complex(-0.3, 0.2)    # Tweak this for different shapes
MAX_ITER  = 300

# Artistic parameters
ARTISTIC_ALPHA  = 1.8    # Adjust glow effect intensity

def hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
    """Convert HSV to RGB"""
    h, s, v  = hsv.T
    r  = v * (1 - s)
    g  = v * ((h % 1 / 6) + 0.5)
    b  = v * min(s, 1)
    return (r, g, b).reshape(-1, 3)

# Generate grid of complex points
x  = np.linspace(X_RANGE[0], X_RANGE[1], WIDTH)
y  = np.linspace(Y_RANGE[0], Y_RANGE[1], HEIGHT)
X, Y  = np.meshgrid(x, y)
Z  = X + 1j * Y

# Initialize iteration counts and mask
div_iter  = np.zeros(Z.shape, dtype=int)
mask  = np.ones(Z.shape, dtype=bool)

# Iterate and record divergence
for i in range(MAX_ITER):
    Z[mask]  = Z[mask]**2 + C
    mask_new  = np.abs(Z) <= 2
    div_iter[~mask & mask_new]  = i
    mask  = mask_new

# Smooth coloring
with np.errstate(divide='warn', invalid='warn'):
    smooth  = div_iter + 1 - np.log(np.log2(np.abs(Z)))
    smooth  = np.nan_to_num(smooth)
smooth_norm  = (smooth - smooth.min()) / (smooth.max() - smooth.min())

# Build HSV image
hsv  = np.zeros((HEIGHT, WIDTH, 3), dtype=float)
hsv[..., 0]  = smooth_norm / 3    # Hue
hsv[..., 1]  = 0.8 + 0.4 * smooth_norm    # Saturation
hsv[..., 2]  = smooth_norm ** 0.3    # Value

# Convert to RGB and apply glow effect
rgb  = (hsv_to_rgb(hsv) * 255).astype(np.uint8)
img  = Image.fromarray(rgb)

# Artistic postprocessing: blur, glow, and enhancement
blur  = img.filter(ImageFilter.GaussianBlur(radius=7))
glow  = Image.blend(img, blur, alpha=ARTISTIC_ALPHA)    # Increased the alpha value for a stronger glow effect
enhanced_img  = ImageEnhance.Brightness(glow).enhance(1.5)

# Save the enhanced image
enhanced_img.save('enhanced_image.png')

print("Image saved as 'enhanced_image.png'")
import os; print("Current directory:", os.getcwd())

