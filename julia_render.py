import numpy as np
import matplotlib.pyplot as plt

width, height = 1600, 1600
x_range = (-1.98, 1.98)
y_range = (-1.98, 1.98)
c = complex(-0.73, 0.22)
max_iter = 300

x = np.linspace(x_range[0], x_range[1], width, dtype=np.float32)
y = np.linspace(y_range[0], y_range[1], height, dtype=np.float32)
X, Y = np.meshgrid(x, y)
Z = X + 1j * Y

iteration = np.full(Z.shape, max_iter, dtype=np.uint16)
mask = np.ones(Z.shape, dtype=bool)
escape_radius = 4.0

for i in range(max_iter):
    if not np.any(mask):
        break
    Z[mask] = Z[mask] ** 2 + c
    mask_new = np.abs(Z) <= escape_radius
    iteration[mask & ~mask_new] = i
    mask = mask_new

with np.errstate(divide='ignore', invalid='ignore'):
    smooth = iteration + 1 - np.log(np.log2(np.abs(Z)))
    smooth = np.nan_to_num(smooth)

fig, ax = plt.subplots(figsize=(8, 8), dpi=112)
im = ax.imshow(smooth, extent=(x_range[0], x_range[1], y_range[0], y_range[1]), 
               origin='lower', cmap='plasma')
ax.set_title('Julia Set (Optimized)', fontsize=14)
ax.set_xlabel('Re(z)', fontsize=12)
ax.set_ylabel('Im(z)', fontsize=12)

plt.tight_layout()
plt.savefig('julia_output.jpg', dpi=112, bbox_inches='tight')
plt.close() 