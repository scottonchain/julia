import numpy as np
import matplotlib.pyplot as plt
import random

width, height = 1600, 1600
x_range = (-2.08, 2.08)
y_range = (-2.0, 2.0)
c = complex(-0.75, 0.29)
max_iter = 300

# Randomize fractal type and iteration formula
random.seed(42)
fractal_types = ['julia', 'mandelbrot', 'burning_ship', 'tricorn', 'phoenix']
selected_type = random.choice(fractal_types)

x = np.linspace(x_range[0], x_range[1], width)
y = np.linspace(y_range[0], y_range[1], height)
X, Y = np.meshgrid(x, y)
Z = X + 1j * Y

iteration = np.zeros(Z.shape, dtype=int)
mask = np.ones(Z.shape, dtype=bool)

if selected_type == 'julia':
    C = c
elif selected_type == 'mandelbrot':
    C = Z.copy()
elif selected_type == 'burning_ship':
    C = Z.copy()
elif selected_type == 'tricorn':
    C = c
elif selected_type == 'phoenix':
    C = Z.copy()
    p = random.uniform(-0.5, 0.5) + 1j * random.uniform(-0.5, 0.5)
    Zold = np.zeros_like(Z)

for i in range(max_iter):
    if selected_type == 'julia':
        Z[mask] = Z[mask] ** 2 + C
    elif selected_type == 'mandelbrot':
        Z[mask] = Z[mask] ** 2 + C[mask]
    elif selected_type == 'burning_ship':
        Z[mask] = (np.abs(Z[mask].real) + 1j * np.abs(Z[mask].imag)) ** 2 + C[mask]
    elif selected_type == 'tricorn':
        Z[mask] = np.conj(Z[mask]) ** 2 + C
    elif selected_type == 'phoenix':
        Z[mask], Zold[mask] = Z[mask] ** 2 + C[mask] + p * Zold[mask], Z[mask]
    
    mask_new = np.abs(Z) <= 2
    iteration[mask & ~mask_new] = i
    mask = mask_new

fig, ax = plt.subplots(figsize=(8, 8), dpi=112)
im = ax.imshow(iteration, extent=(x_range[0], x_range[1], y_range[0], y_range[1]), 
               origin='lower', cmap='plasma')
ax.set_title(f'Fractal (Random Type: {selected_type})', fontsize=14)
ax.set_xlabel('Re(z)', fontsize=12)
ax.set_ylabel('Im(z)', fontsize=12)

plt.tight_layout()
plt.savefig('julia_output.jpg', dpi=112, bbox_inches='tight')
plt.close() 