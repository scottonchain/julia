import numpy as np
import matplotlib.pyplot as plt

width, height = 900, 900
x_range = (-1.11, 1.11)
y_range = (-0.8, 0.8)
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

fig, ax = plt.subplots(figsize=(8, 8), dpi=112)
im = ax.imshow(iteration, extent=(x_range[0], x_range[1], y_range[0], y_range[1]), 
               origin='lower', cmap='rainbow')
ax.set_title('Julia Set (Bright Rainbow)', fontsize=14)
ax.set_xlabel('Re(z)', fontsize=12)
ax.set_ylabel('Im(z)', fontsize=12)

plt.tight_layout()
plt.savefig('julia_output.jpg', dpi=112, bbox_inches='tight')
plt.close() 