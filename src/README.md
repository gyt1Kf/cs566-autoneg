# Source Code

### Installation

From the project root, run:

python -m pip install -r requirements.txt

## DNG to PNG Conversion

`dng_to_png.py` converts DNG negative scans into smaller 16-bit RGB PNG
images for development and testing. It preserves the aspect ratio,
does not upscale smaller images, and leaves the original DNG files unchanged.
The default maximum side length is 2000 pixels.

Options:

1. Limit the longest side to 1600 pixels:

python src/dng_to_png.py --max-side 1600

2. Specify custom input and output directories:

python src/dng_to_png.py --input data/my_dng --output data/my_png --max-side 2000

3. Replace existing output files:

python src/dng_to_png.py --overwrite

## Negative Inversion

`negative_inversion.py` converts film negatives into positive images using an algorithm adapted from 
darktable’s negadoctor. It estimates the film-base color, compensates for the orange mask, and 
adjusts the density range. Further color and exposure correction can be applied afterward.

Install dependencies (Python 3.10+):
pip install numpy pillow opencv-python-headless

Run from the src/ directory:

python negative_inversion.py \
  ../data/samples/negative/19930125-05-hk.png \
  ../data/samples/inversion_output/19930125-05-hk.png \
  --encoding srgb

Supports 8-bit and 16-bit RGB PNG input. Use --encoding linear if the input was exported in linear RGB.
Output .png for a preview or .npy to preserve floating-point data for subsequent processing.

Film-base color is estimated automatically. For more reliable results, specify an unexposed film region
with --base-roi X Y WIDTH HEIGHT.

Show all options:
python negative_inversion.py --help

Adapted from darktable’s negadoctor, licensed under GPL-3.0-or-later.
