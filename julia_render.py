import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from matplotlib.colors import hsv_to_rgb

width, height = 1600, 1600
x_range = (-0.78, 0.78)
y_range = (-0.61, 0.61)
c = complex(-0.69, 0.24)
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

# Apply scipy filters for enhancement
iteration_smooth = ndimage.gaussian_filter(iteration.astype(float), sigma=0.5)
iteration_enhanced = ndimage.uniform_filter(iteration_smooth, size=2)

# Smooth coloring
with np.errstate(divide='ignore', invalid='ignore'):
    smooth = iteration_enhanced + 1 - np.log(np.log2(np.abs(Z)))
    smooth = np.nan_to_num(smooth)
smooth_norm = (smooth - smooth.min()) / (smooth.max() - smooth.min())

# Bright palette
hsv = np.zeros((height, width, 3), dtype=float)
hsv[..., 0] = (0.6 * smooth_norm + 0.3) % 1
hsv[..., 1] = 0.95 - 0.1 * np.abs(np.sin(2 * np.pi * smooth_norm))
hsv[..., 2] = smooth_norm ** 0.2

rgb = (hsv_to_rgb(hsv) * 255).astype(np.uint8)

fig, ax = plt.subplots(figsize=(8, 8), dpi=112)
im = ax.imshow(rgb, extent=(x_range[0], x_range[1], y_range[0], y_range[1]), 
               origin='lower')
ax.set_title('Julia Set (SciPy Enhanced)', fontsize=14)
ax.set_xlabel('Re(z)', fontsize=12)
ax.set_ylabel('Im(z)', fontsize=12)
ax.grid(True, color='white', alpha=0.3, linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.savefig('julia_output.jpg', dpi=112, bbox_inches='tight')
plt.close() 