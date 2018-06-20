import numpy as np
import matplotlib.pyplot as plt

width, height = 1600, 1600
x_range = (-2.0, 2.0)
y_range = (-2.0, 2.0)
c = complex(-0.68, 0.22)
max_iter = 300

x = np.linspace(x_range[0], x_range[1], width, dtype=np.float32)
y = np.linspace(y_range[0], y_range[1], height, dtype=np.float32)
X, Y = np.meshgrid(x, y)
Z = X + 1j * Y

iteration = np.full(Z.shape, max_iter, dtype=np.uint16)
mask = np.ones(Z.shape, dtype=bool)
escape_radius = 4.0
escape_radius_sq = escape_radius * escape_radius

Z_sq = np.zeros_like(Z, dtype=np.complex64)
Z_abs_sq = np.zeros(Z.shape, dtype=np.float32)

for i in range(max_iter):
    if not np.any(mask):
        break
    
    Z_sq[mask] = Z[mask] * Z[mask]
    Z[mask] = Z_sq[mask] + c
    
    Z_abs_sq[mask] = Z[mask].real * Z[mask].real + Z[mask].imag * Z[mask].imag
    mask_new = Z_abs_sq <= escape_radius_sq
    iteration[mask & ~mask_new] = i
    mask = mask_new

with np.errstate(divide='ignore', invalid='ignore'):
    smooth = iteration + 1 - np.log(np.log2(np.sqrt(Z_abs_sq)))
    smooth = np.nan_to_num(smooth)

fig, ax = plt.subplots(figsize=(8, 8), dpi=112)
im = ax.imshow(smooth, extent=(x_range[0], x_range[1], y_range[0], y_range[1]), 
               origin='lower', cmap='plasma')
ax.set_title('Julia Set (Highly Optimized)', fontsize=14)
ax.set_xlabel('Re(z)', fontsize=12)
ax.set_ylabel('Im(z)', fontsize=12)

plt.tight_layout()
plt.savefig('julia_output.jpg', dpi=112, bbox_inches='tight')
plt.close() 