import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from matplotlib.colors import hsv_to_rgb

width, height = 1600, 1600
x_range = (-0.62, 0.62)
y_range = (-0.62, 0.62)
c = complex(-0.71, 0.28)
max_iter = 500

@jit(nopython=True)
def julia_numba(x_range, y_range, width, height, c, max_iter):
    x = np.linspace(x_range[0], x_range[1], width)
    y = np.linspace(y_range[0], y_range[1], height)
    
    result = np.zeros((height, width), dtype=np.int32)
    
    for i in range(height):
        for j in range(width):
            z = complex(x[j], y[i])
            for k in range(max_iter):
                z = z * z + c
                if abs(z) > 2:
                    result[i, j] = k
                    break
            else:
                result[i, j] = max_iter
    
    return result

# Generate fractal using Numba
iteration = julia_numba(x_range, y_range, width, height, c, max_iter)

# Smooth coloring
with np.errstate(divide='ignore', invalid='ignore'):
    smooth = iteration + 1 - np.log(np.log2(np.abs(complex(x_range[0], y_range[0]))))
    smooth = np.nan_to_num(smooth)
smooth_norm = (smooth - smooth.min()) / (smooth.max() - smooth.min())

# Bright palette
hsv = np.zeros((height, width, 3), dtype=float)
hsv[..., 0] = (0.7 * smooth_norm + 0.2) % 1
hsv[..., 1] = 0.95 - 0.1 * np.abs(np.sin(2 * np.pi * smooth_norm))
hsv[..., 2] = smooth_norm ** 0.2

rgb = (hsv_to_rgb(hsv) * 255).astype(np.uint8)

fig, ax = plt.subplots(figsize=(8, 8), dpi=112)
im = ax.imshow(rgb, extent=(x_range[0], x_range[1], y_range[0], y_range[1]), 
               origin='lower')
ax.set_title('Julia Set (Numba Accelerated)', fontsize=14)
ax.set_xlabel('Re(z)', fontsize=12)
ax.set_ylabel('Im(z)', fontsize=12)
ax.grid(True, color='white', alpha=0.3, linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.savefig('julia_output.jpg', dpi=112, bbox_inches='tight')
plt.close() 