import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

# Parameters for a scientific Julia set plot
width, height = 1600, 1600
x_range = (-1.51, 1.51)
y_range = (-1.51, 1.51)
c = complex(-0.72, 0.27)
max_iter = 400

# Generate grid
x = np.linspace(x_range[0], x_range[1], width)
y = np.linspace(y_range[0], y_range[1], height)
X, Y = np.meshgrid(x, y)
Z = X + 1j * Y

# Julia set calculation
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

# HSV colormap for scientific look
hsv = np.zeros((height, width, 3), dtype=float)
hsv[..., 0] = (0.6 * smooth_norm + 0.2) % 1
hsv[..., 1] = 0.8
hsv[..., 2] = smooth_norm ** 0.8
rgb = hsv_to_rgb(hsv)

# Plot with scientific style
fig, ax = plt.subplots(figsize=(8, 8), dpi=112)
im = ax.imshow(rgb, extent=(x_range[0], x_range[1], y_range[0], y_range[1]), origin='lower')
ax.set_title('Julia Set (c = -0.7 + 0.27015i)', fontsize=14)
ax.set_xlabel('Re(z)', fontsize=12)
ax.set_ylabel('Im(z)', fontsize=12)
ax.grid(True, color='white', alpha=0.2, linestyle='--', linewidth=0.5)

# Add colorbar
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
sm = ScalarMappable(cmap='hsv', norm=Normalize(vmin=smooth_norm.min(), vmax=smooth_norm.max()))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Normalized Iteration (Smooth)', fontsize=12)

# Save as scientific-looking chart
plt.tight_layout()
plt.savefig('julia_output.jpg', dpi=112, bbox_inches='tight')
plt.close() 