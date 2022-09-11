import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from matplotlib.colors import hsv_to_rgb

width, height = 1600, 1600
x_range = (-1.35, 1.35)
y_range = (-1.15, 1.15)
c = complex(-0.39, 0.6)
max_iter = 360

x = np.linspace(x_range[0], x_range[1], width)
y = np.linspace(y_range[0], y_range[1], height)
X, Y = np.meshgrid(x, y)
Z = X + 1j * Y

div_iter = np.zeros(Z.shape, dtype=int)
mask = np.ones(Z.shape, dtype=bool)
for i in range(max_iter):
    Z[mask] = Z[mask] ** 2 + c
    mask_new = np.abs(Z) <= 2
    div_iter[mask & ~mask_new] = i
    mask = mask_new

with np.errstate(divide='ignore', invalid='ignore'):
    smooth = div_iter + 1 - np.log(np.log2(np.abs(Z)))
    smooth = np.nan_to_num(smooth)
smooth_norm = (smooth - smooth.min()) / (smooth.max() - smooth.min())

# Fire palette
hsv = np.zeros((height, width, 3), dtype=float)
hsv[..., 0] = (0.05 + 0.1 * smooth_norm) % 1
hsv[..., 1] = 1.0 - 0.5 * smooth_norm
hsv[..., 2] = smooth_norm ** 0.7

rgb = (hsv_to_rgb(hsv) * 255).astype(np.uint8)
img = Image.fromarray(rgb)

img = img.filter(ImageFilter.GaussianBlur(radius=8))

def horizontal_ripple(im, amp=10, freq=0.1):
    arr = np.array(im)
    for i in range(arr.shape[0]):
        arr[i] = np.roll(arr[i], int(amp * np.sin(freq * i)))
    return Image.fromarray(arr)

img = horizontal_ripple(img, amp=20, freq=0.18)
img = ImageEnhance.Contrast(img).enhance(1.5)

output_path = 'julia_output.jpg'
img.save(output_path) 