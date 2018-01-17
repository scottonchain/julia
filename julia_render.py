import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
from matplotlib.colors import hsv_to_rgb

width, height = 1600, 1600
x_range = (-1.69, 1.69)
y_range = (-1.75, 1.75)
c = complex(0.35, -0.18)
max_iter = 380

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

hsv = np.zeros((height, width, 3), dtype=float)
hsv[..., 0] = (0.8 * smooth_norm + 0.3) % 1
hsv[..., 1] = 0.9 - 0.7 * smooth_norm
hsv[..., 2] = smooth_norm ** 0.7

rgb = (hsv_to_rgb(hsv) * 255).astype(np.uint8)
img = Image.fromarray(rgb)

# Shuffle in blocks
def shuffle_blocks(im, block=40):
    arr = np.array(im)
    blocks = []
    for i in range(0, arr.shape[0], block):
        for j in range(0, arr.shape[1], block):
            blocks.append(arr[i:i+block, j:j+block].copy())
    np.random.shuffle(blocks)
    idx = 0
    for i in range(0, arr.shape[0], block):
        for j in range(0, arr.shape[1], block):
            arr[i:i+block, j:j+block] = blocks[idx]
            idx += 1
    return Image.fromarray(arr)

img = shuffle_blocks(img, block=60)
img = ImageOps.posterize(img, 2)
img = ImageEnhance.Color(img).enhance(2.0)

output_path = 'julia_output.jpg'
img.save(output_path) 