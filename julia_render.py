import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

width, height = 1600, 1600
x_range = (-1.1, 1.1)
y_range = (-0.82, 0.82)
c = complex(-0.73, 0.32)
max_iter = 300

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

# Smooth coloring
with np.errstate(divide='ignore', invalid='ignore'):
    smooth = iteration + 1 - np.log(np.log2(np.abs(Z)))
    smooth = np.nan_to_num(smooth)
smooth_norm = (smooth - smooth.min()) / (smooth.max() - smooth.min())

# Bright autumn palette
from matplotlib import cm
rgb = cm.autumn(smooth_norm)[..., :3]

fig, ax = plt.subplots(figsize=(8, 8), dpi=112)
im = ax.imshow(rgb, extent=(x_range[0], x_range[1], y_range[0], y_range[1]), 
               origin='lower')
ax.set_title('Julia Set (Bright Autumn)', fontsize=14)
ax.set_xlabel('Re(z)', fontsize=12)
ax.set_ylabel('Im(z)', fontsize=12)

plt.tight_layout()
plt.savefig('julia_output.jpg', dpi=112, bbox_inches='tight')
plt.close() 