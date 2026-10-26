# Background remover with OpenCV GrabCut

Removes the background from product photos and writes transparent PNGs, or JPGs on pure white for marketplace
main images. Uses OpenCV's GrabCut algorithm: no model download, no GPU, two dependencies.

## How it works

```
photo ──► border color (median in Lab) ──► initial mask: frame = background,
          similar to border = probable background, different = probable foreground
      ──► cv2.grabCut (color models + graph cut) on a copy of at most 1024 px
      ──► opening/closing, fill holes inside the product, optional "largest object only"
      ──► scale mask to full size, soft edge ──► PNG with alpha  or  JPG on white
```

GrabCut needs a first guess of foreground and background. Product photos are usually shot on a plain, light
background, so the image border is a good estimate of the background color. Light parts inside the product
(labels, reflections) that look like the background are restored by hole filling.

## Setup and run

```bash
pip install -r requirements.txt
cd src
python -m bgremove ../photos ../out              # transparent PNGs
python -m bgremove ../photos ../out --white      # JPGs on white
python -m bgremove ../photos ../out --single --tolerance 20
```

Each processed file prints its foreground share, for example `mug.jpg -> mug.png (23% foreground)`. Values near 0 %
or 100 % usually mean the cut-out failed.

In Python:

```python
import cv2
from bgremove.remover import remove_background
cv2.imwrite("out.png", remove_background(cv2.imread("photo.jpg")))
```

Tests (synthetic product photos, no files needed): `pytest`.

## Configuration (`Settings`)

| Field | Default | Meaning |
|---|---|---|
| `border` | 0.02 | share of width/height used as sure background |
| `color_tolerance` | 30 | Lab distance from the background color that counts as "different" |
| `iterations` | 5 | GrabCut iterations |
| `feather` | 3 | soft edge in pixels (0 = hard edge) |
| `keep_largest` | False | keep only the largest object |
| `fill_holes` | True | fill enclosed background regions inside the product |
| `max_side` | 1024 | GrabCut runs on a copy of at most this size |

## Limitations

- Needs a fairly plain background that differs from the product. A white product on white, glass and
  transparent objects, or busy backgrounds give poor masks.
- The product must not touch the image border (the border is treated as background).
- Hole filling also fills real holes (for example the inside of a mug handle); turn it off with
  `Settings(fill_holes=False)` for such products.
- Fine structures such as hair, fur or thin cables are not handled well.

## For production

- For difficult photos use a neural model such as [rembg](https://github.com/danielgatis/rembg) (downloads a
  U2-Net/BiRefNet model) and keep GrabCut as a fast, dependency-free path for studio shots.
- Standardize the photo setup (background, light, margin): it improves results more than parameter tuning.
- Check the marketplace image rules (for example pure white background, product share of the frame) and add an
  automatic crop and padding step.
