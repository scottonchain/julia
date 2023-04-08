import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw
from matplotlib.colors import hsv_to_rgb

width, height = 1600, 1600
x_range = (-1.44, 1.44)
y_range = (-1.44, 1.44)
c = complex(-0.66, -0.35)
max_iter = 340

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
hsv[..., 0] = (0.85 * smooth_norm + 0.15) % 1
hsv[..., 1] = 1.0 - 0.5 * smooth_norm
hsv[..., 2] = smooth_norm ** 0.7

rgb = (hsv_to_rgb(hsv) * 255).astype(np.uint8)
img = Image.fromarray(rgb)

# Pixel sorting effect
def pixel_sort(im):
    arr = np.array(im)
    for col in arr.transpose(1,0,2):
        col.sort(axis=0)
    return Image.fromarray(arr)

img = pixel_sort(img)

def add_stripes(im, stripe_width=20):
    draw = ImageDraw.Draw(im)
    for x in range(0, im.width, stripe_width*2):
        draw.rectangle([x, 0, x+stripe_width, im.height], fill=(255,255,255,40))
    return im

img = add_stripes(img, stripe_width=25)
img = ImageEnhance.Brightness(img).enhance(1.3)

output_path = 'julia_output.jpg'
img.save(output_path) 