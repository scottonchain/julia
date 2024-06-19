import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# Define the Julia set parameters
x_min, x_max = -1.5, 1.5
y_min, y_max = -1.5, 1.5
width, height = 800, 800
scale_x, scale_y = width / (x_max - x_min), height / (y_max - y_min)

# Generate the Julia set
def julia_set(x, y):
    c_real, c_imag = 0.285d + 0.01j, 0.02d + 0.03j
    return np.where((x - c_real) ** 2 + (y - c_imag) ** 2 > 4, 1, 0)

# Create the grid of points to evaluate the Julia set function
x = np.linspace(x_min, x_max, width)
y = np.linspace(y_min, y_max, height)
X, Y = np.meshgrid(x, y)

# Evaluate the Julia set function on each point in the grid
Z = julia_set(X, Y)

# Add an artistic effect: blur and contrast adjustment
enhanced = gaussian_filter(Z, sigma=2)  # Blur with a small radius (sigma)
enhanced[enhanced > 0.5] = 1  # Contrast adjustment: set all values above 0.5 to 1

# Display the enhanced Julia set
plt.figure(figsize=(6, 6))
plt.axis('off')
plt.imshow(enhanced, cmap='gray', vmin=0, vmax=1)
plt.show()

