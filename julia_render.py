import numpy as np
import matplotlib.pyplot as plt

width, height = 1600, 1600
x_range = (-2.04, 2.04)
y_range = (-2.04, 2.04)
c = complex(-0.68, 0.31)
max_iter = 300

# Naive implementation with nested loops
result = np.zeros((height, width))
for i in range(height):
    for j in range(width):
        x = x_range[0] + (x_range[1] - x_range[0]) * j / width
        y = y_range[0] + (y_range[1] - y_range[0]) * (height - 1 - i) / height
        z = complex(x, y)
        
        for k in range(max_iter):
            z = z * z + c
            if abs(z) > 2:
                result[i, j] = k
                break
        else:
            result[i, j] = max_iter

fig, ax = plt.subplots(figsize=(8, 8), dpi=112)
im = ax.imshow(result, extent=(x_range[0], x_range[1], y_range[0], y_range[1]), 
               origin='lower', cmap='plasma')
ax.set_title('Julia Set (Naive)', fontsize=14)
ax.set_xlabel('Re(z)', fontsize=12)
ax.set_ylabel('Im(z)', fontsize=12)

plt.tight_layout()
plt.savefig('julia_output.jpg', dpi=112, bbox_inches='tight')
plt.close() 