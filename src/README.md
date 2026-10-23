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
