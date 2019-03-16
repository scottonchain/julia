import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

width, height = 1600, 1600
# Zoomed in region for detail
x_range = (-2.05, 2.05)
y_range = (-2.05, 2.05)
c = complex(0.24, 0.01)
max_iter = 600

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

# Smooth coloring for better detail
with np.errstate(divide='ignore', invalid='ignore'):
    smooth = iteration + 1 - np.log(np.log2(np.abs(Z)))
    smooth = np.nan_to_num(smooth)

# Use bright, warm prismatic colormap
fig, ax = plt.subplots(figsize=(8, 8), dpi=112)
im = ax.imshow(smooth, extent=(x_range[0], x_range[1], y_range[0], y_range[1]), 
               origin='lower', cmap='viridis', interpolation='bilinear')
ax.set_title('Julia Set Detail (c = 0.285 + 0.01i)', fontsize=14)
ax.set_xlabel('Re(z)', fontsize=12)
ax.set_ylabel('Im(z)', fontsize=12)
ax.grid(True, color='white', alpha=0.3, linestyle='--', linewidth=0.5)

# Add colorbar
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Iteration Count (Smooth)', fontsize=12)

plt.tight_layout()
plt.savefig('julia_output.jpg', dpi=112, bbox_inches='tight')
plt.close() 