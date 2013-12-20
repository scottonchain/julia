import numpy as np
import matplotlib.pyplot as plt
import random

width, height = 900, 900
c = complex(-0.7, 0.27015)
max_iter = 300

# Randomize zoom level and viewport
random.seed(42)
zoom_factor = random.uniform(0.1, 2.0)
center_x = random.uniform(-1.0, 1.0)
center_y = random.uniform(-1.0, 1.0)

x_range = (center_x - zoom_factor, center_x + zoom_factor)
y_range = (center_y - zoom_factor, center_y + zoom_factor)

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

fig, ax = plt.subplots(figsize=(8, 8), dpi=112)
im = ax.imshow(iteration, extent=(x_range[0], x_range[1], y_range[0], y_range[1]), 
               origin='lower', cmap='plasma')
ax.set_title(f'Julia Set (Random Zoom: center=({center_x:.2f},{center_y:.2f}), zoom={zoom_factor:.2f})', fontsize=12)
ax.set_xlabel('Re(z)', fontsize=12)
ax.set_ylabel('Im(z)', fontsize=12)

plt.tight_layout()
plt.savefig('julia_output.jpg', dpi=112, bbox_inches='tight')
plt.close() 