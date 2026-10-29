# AutoNeg

CS 566 project: automatic conversion of scanned color negative film into normal photos.

Pipeline:
1. Negative inversion + film base (orange mask) compensation, adapted from darktable's negadoctor
2. Automatic color and exposure correction
3. Dust and scratch detection and removal (in progress)

## Setup

```
python -m pip install -r requirements.txt
```

## Usage

Run the full pipeline on a scan (from `src/`):

```
python main.py ../data/samples/negative/19930125-05-hk.png ../results/hk.png
```

This saves the corrected image and a `_inverted.npy` file with the linear inverted image.
Input is read as linear by default, which is what `dng_to_png.py` outputs. For a normal sRGB scan (e.g. a JPEG from a scanner) add `--encoding srgb`.
If the scan has unexposed film border, pass it with `--base-roi X Y W H` for a better film base estimate.

Individual steps:

- `dng_to_png.py` - converts DNG scans to 16-bit PNGs (longest side 2000 px by default, change with `--max-side`)
- `negative_inversion.py` - inversion only. Save as `.npy` to keep float precision, `.png` for a preview
- `color_correction.py` - color/exposure correction on an inverted image
- `evaluate.py` - runs color correction on BlueNeg and reports PSNR, SSIM, CIEDE2000 and a/b error against the pseudo ground truth

Run any script with `--help` for all options.

## Data

Datasets are stored locally and not committed, except `data/samples/`.

- BlueNeg (negative conversion, color): put it in `data/blueneg/`. Sample images are copyright Tien-Tsin Wong, see `data/samples/LICENSE`
- Authentically Damaged & Manually Restored Film Scans (dust and scratches)
- FilmDamageSimulator (synthetic damage with masks)

Outputs go in `results/` (also not committed).

## License

`negative_inversion.py` is adapted from darktable's [negadoctor](https://github.com/darktable-org/darktable/blob/master/src/iop/negadoctor.c) and is GPL-3.0-or-later.
