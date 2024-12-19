
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, filters

# Define the Julia set parameters
x_min, x_max = -1.5, 1.5
y_min, y_max = -1.5, 1.5
width, height = 800, 800
scale_x, scale_y = width / (x_max - x_min), height / (y_max - y_min)

# Generate the Julia set
def julia_set(x, y):
    c_real, c_imag = 0.285 + 0.01j, 0.02 + 0.03j
    return np.where((x - c_real) ** 2 + (y - c_imag) ** 2 > 4, 1, 0)

# Create the grid of points to evaluate the Julia set function
x = np.linspace(x_min, x_max, width)
y = np.linspace(y_min, y_max, height)
X, Y = np.meshgrid(x, y)

# Evaluate the Julia set function on each point in the grid
Z = julia_set(X, Y)

# Add artistic effects:
enhanced = Z.copy()  # Create a copy of the original image

# Blur with Gaussian filter (adjust sigma for different blur levels)
sigma = 2.0
enhanced = gaussian_filter(enhanced, sigma=sigma) * 255  # Scale to [0-255]

# Apply contrast adjustment:
contrast_factor = 1.5  # Adjust this value for desired contrast level
threshold = np.percentile(enhanced, 95)
enhanced[enhanced < threshold] *= contrast_factor

# Add a subtle gradient effect:
gradient_strength = 10  # Adjust this value for different gradient levels
for i in range(len(enhanced)):
    for j in range(len(enhanced[0])):
        if enhanced[i][j] > threshold:  # Apply the gradient only to areas with high contrast
            r, g, b = int((1 - (enhanced[i][j] / 255)) * 256), int((1 - (enhanced[i][j] / 255)) * 128), int((1 - (enhanced[i][j] / 255)) * 64)
            enhanced[i][j] = np.array([r, g, b]) / 255

# Add a subtle noise effect:
noise_level = 0.05
for i in range(len(enhanced)):
    for j in range(len(enhanced[0])):
        if random.random() < noise_level:  # Randomly add noise to some pixels
            enhanced[i][j] += np.array([random.randint(-10, 10), random.randint(-10, 10), random.randint(-10, 10)]) / 255

# Display the enhanced Julia set:
plt.figure(figsize=(6, 6))
plt.axis('off')
plt.imshow(enhanced.astype(np.uint8), cmap='gray', vmin=0, vmax=1)  # Convert to uint8 for display
plt.show()

