# Pure Python Image Processor

A small image processor built with **only core Python** (no NumPy, OpenCV, or Pillow).
A grayscale image is a list of rows, each row a list of ints from 0 to 255.
Every function takes an image and returns a **new** one; the input is never modified
(the demo checks this at the end).

## Run it

```bash
python3 imgproc.py
```

Python 3.10+, no dependencies. The full output is also saved in `output.txt` and reproduced below.

## The test image

A 5x5 image: dark left side (10), bright right side (200), and one mid-gray pixel (120)
in row 1. The vertical edge and the lone pixel make every operation easy to check by eye.

## Functions

### Pixel-wise (each output pixel depends only on the same input pixel)

| Function | What it does | What the output shows |
|---|---|---|
| `invert(img)` | `p -> 255 - p` | 10 becomes 245, 200 becomes 55, 120 becomes 135 |
| `adjust_brightness(img, delta)` | adds `delta`, clamps to 0-255 | `+60`: 200 saturates at 255. `-60`: 10 clamps at 0 |
| `threshold(img, cutoff)` | 255 if `p >= cutoff` else 0 | cutoff 100 keeps the bright side and the 120 spot, zeroes the rest |

### Geometric (pixels move, values unchanged)

| Function | What it does | What the output shows |
|---|---|---|
| `flip_horizontal(img)` | mirrors left/right | bright side moves to the left (200 200 10 10 10); the 120 spot stays in column 2 because the image is 5 wide and column 2 is the middle |
| `flip_vertical(img)` | mirrors top/bottom | the 120 spot moves from row 1 to row 3 |
| `rotate_90_cw(img)` | rotates 90 degrees clockwise | a 2x3 image becomes 3x2: `[[1,2,3],[4,5,6]]` becomes `[[4,1],[5,2],[6,3]]` |
| `crop(img, top, left, height, width)` | returns a window; raises `ValueError` if it leaves the image | a 3x3 window starting at row 1, col 2 |

### Neighborhood (each output pixel depends on a 3x3 patch of input)

| Function | What it does | What the output shows |
|---|---|---|
| `convolve(img, kernel, divisor, absolute)` | slides a square kernel over the image | the general engine used by the two below |
| `box_blur(img)` | 3x3 average | the sharp 10 to 200 edge becomes a gradient (10, 22, 86, 149, 200); the 120 spot is smeared into its neighbors |
| `sobel_edges(img)` | `abs(Gx) + abs(Gy)` with Sobel kernels | strong (220-255) along the vertical edge, 0 in flat regions; the edge response is weaker (0 instead of 220) in column 1 of rows 3-4, where the 120 spot is no longer in the 3x3 neighborhood |

### Border handling: replicate (clamp)

Near the edge, a 3x3 kernel needs pixels that don't exist. I chose **replicate**: an
out-of-range coordinate snaps to the nearest edge pixel (`get_pixel_clamped`).

Why:
- **Output stays the same size**, so results can be compared pixel-for-pixel with the input.
- **No fake edges.** Zero padding would pretend a black frame surrounds the image, so Sobel would
  report a strong edge along every border of a bright image. Replicating the edge assumes the image
  continues as it is, so flat borders stay flat. You can see this in the output: the left and right
  columns of the Sobel result are 0.
- Cropping the border (no padding) would shrink the image and lose the outermost pixels, which
  matters for lane and obstacle detection where objects often touch the frame edge.

Trade-off: replication slightly overweights edge pixels in the average.

## Bonus: color and PPM

- `write_ppm(img, path)` / `read_ppm(path)`: hand-written ASCII PPM (P3) reader and writer for RGB
  images stored as a grid of `[r, g, b]` pixels. The demo shows a lossless round trip.
- `to_grayscale(rgb)`: luminance `0.299 R + 0.587 G + 0.114 B`. Green is weighted most because the human eye is most
  sensitive to it, so pure green (150) comes out far brighter than pure blue (29).

## Full terminal output

```text
============================================
invert
============================================
BEFORE:
   10  10  10 200 200
   10  10 120 200 200
   10  10  10 200 200
   10  10  10 200 200
   10  10  10 200 200

AFTER:
  245 245 245  55  55
  245 245 135  55  55
  245 245 245  55  55
  245 245 245  55  55
  245 245 245  55  55

============================================
adjust_brightness (+60)
============================================
BEFORE:
   10  10  10 200 200
   10  10 120 200 200
   10  10  10 200 200
   10  10  10 200 200
   10  10  10 200 200

AFTER:
   70  70  70 255 255
   70  70 180 255 255
   70  70  70 255 255
   70  70  70 255 255
   70  70  70 255 255

============================================
adjust_brightness (-60, clamps at 0)
============================================
BEFORE:
   10  10  10 200 200
   10  10 120 200 200
   10  10  10 200 200
   10  10  10 200 200
   10  10  10 200 200

AFTER:
    0   0   0 140 140
    0   0  60 140 140
    0   0   0 140 140
    0   0   0 140 140
    0   0   0 140 140

============================================
threshold (cutoff 100)
============================================
BEFORE:
   10  10  10 200 200
   10  10 120 200 200
   10  10  10 200 200
   10  10  10 200 200
   10  10  10 200 200

AFTER:
    0   0   0 255 255
    0   0 255 255 255
    0   0   0 255 255
    0   0   0 255 255
    0   0   0 255 255

============================================
flip_horizontal
============================================
BEFORE:
   10  10  10 200 200
   10  10 120 200 200
   10  10  10 200 200
   10  10  10 200 200
   10  10  10 200 200

AFTER:
  200 200  10  10  10
  200 200 120  10  10
  200 200  10  10  10
  200 200  10  10  10
  200 200  10  10  10

============================================
flip_vertical
============================================
BEFORE:
   10  10  10 200 200
   10  10 120 200 200
   10  10  10 200 200
   10  10  10 200 200
   10  10  10 200 200

AFTER:
   10  10  10 200 200
   10  10  10 200 200
   10  10  10 200 200
   10  10 120 200 200
   10  10  10 200 200

============================================
rotate_90_cw (2x3 becomes 3x2)
============================================
BEFORE:
    1   2   3
    4   5   6

AFTER:
    4   1
    5   2
    6   3

============================================
crop (top=1, left=2, height=3, width=3)
============================================
BEFORE:
   10  10  10 200 200
   10  10 120 200 200
   10  10  10 200 200
   10  10  10 200 200
   10  10  10 200 200

AFTER:
  120 200 200
   10 200 200
   10 200 200

============================================
box_blur (3x3, replicate borders)
============================================
BEFORE:
   10  10  10 200 200
   10  10 120 200 200
   10  10  10 200 200
   10  10  10 200 200
   10  10  10 200 200

AFTER:
   10  22  86 149 200
   10  22  86 149 200
   10  22  86 149 200
   10  10  73 137 200
   10  10  73 137 200

============================================
sobel_edges (replicate borders)
============================================
BEFORE:
   10  10  10 200 200
   10  10 120 200 200
   10  10  10 200 200
   10  10  10 200 200
   10  10  10 200 200

AFTER:
    0 220 255 255   0
    0 220 255 255   0
    0 220 255 255   0
    0   0 255 255   0
    0   0 255 255   0

============================================
bonus: PPM round trip + to_grayscale
============================================
RGB written then read back identical: True
Grayscale of the 2x3 RGB image:
   76 150  29
  226 255   0

Original image unchanged by all operations: True
```
