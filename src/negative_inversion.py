# SPDX-License-Identifier: GPL-3.0-or-later
# Adapted from darktable src/iop/negadoctor.c.
# Original algorithm: Copyright (C) 2020-2026 darktable developers.
"""

AutoNeg: film-base compensation and negative inversion (Python 3.10+).

INSTALL
    pip install numpy pillow opencv-python-headless

COMMAND LINE
    python negative_inversion.py --help
    python negative_inversion.py scan.png positive.npy --encoding srgb
    python negative_inversion.py scan.png positive.npy --encoding srgb \
        --base-roi 10 20 100 80
    python negative_inversion.py scan.npy positive.npy --encoding linear \
        --dmin 1.0 0.45 0.25 --dmax 2.0 --offset 0.0
    python negative_inversion.py scan.png preview.png --encoding srgb

TEAM API / INTERFACE
    params = estimate_parameters(negative_linear, base_roi=(x, y, w, h))
    positive_linear = invert_negative(negative_linear, params)
    # Next stage: correct_color_exposure(positive_linear)

    Input: H x W x 3 RGB NumPy array, finite, nonnegative, LINEAR-light.
    Use a consistent RGB color space for the image and Dmin. Typical input
    scale is 0..1; values above 1 are supported. Dmin uses the same scale.
    Output: same shape, float32, linear inversion intermediate, range [0, 1).
    No white balance, paper gamma, exposure, or highlight compression is
    applied. This is an intermediate, not a calibrated scene-radiance image.
    .npy output preserves float data for the next stage. PNG is an 8-bit
    sRGB preview only; do not use it as the pipeline interchange format.

SOURCE / LICENSE
    https://github.com/darktable-org/darktable/blob/master/src/iop/negadoctor.c
    Derived from the user-supplied source; GPL-3.0-or-later, without warranty.
    License text: https://www.gnu.org/licenses/gpl-3.0.html
"""

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
import sys
import warnings

import numpy as np

THRESHOLD = 2.0 ** -32
ROI = tuple[int, int, int, int]


@dataclass(frozen=True)
class InversionParameters:
    """Dmin: linear RGB film base; dmax: log10 density range; offset: bias."""

    dmin: tuple[float, float, float]
    dmax: float
    offset: float = 0.0


def _validate_image(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image, dtype=np.float64)
    if array.ndim != 3 or array.shape[2] != 3 or array.size == 0:
        raise ValueError("Input must be a nonempty H x W x 3 RGB array.")
    if not np.all(np.isfinite(array)) or np.any(array < 0):
        raise ValueError("Input pixels must be finite and nonnegative.")
    return array


def _region(image: np.ndarray, roi: ROI | None) -> np.ndarray:
    if roi is None:
        return image
    if len(roi) != 4 or any(int(v) != v for v in roi):
        raise ValueError("ROI must contain four integers: x y width height.")
    x, y, width, height = map(int, roi)
    if min(x, y) < 0 or min(width, height) <= 0:
        raise ValueError("ROI origin must be nonnegative; size must be positive.")
    if x + width > image.shape[1] or y + height > image.shape[0]:
        raise ValueError("ROI extends beyond the image boundaries.")
    return image[y:y + height, x:x + width]


def _validate_parameters(params: InversionParameters) -> np.ndarray:
    base = np.asarray(params.dmin, dtype=np.float64)
    if base.shape != (3,) or not np.all(np.isfinite(base)) or np.any(base <= 0):
        raise ValueError("Dmin must contain three finite, positive linear RGB values.")
    if not np.isfinite(params.dmax) or params.dmax <= 0:
        raise ValueError("Dmax must be finite and greater than zero.")
    if not np.isfinite(params.offset):
        raise ValueError("Offset must be finite.")
    return base


def estimate_parameters(
    negative_linear: np.ndarray,
    *,
    dmin: tuple[float, float, float] | None = None,
    base_roi: ROI | None = None,
    range_roi: ROI | None = None,
    dmax: float | None = None,
    offset: float | None = None,
    low_percentile: float = 0.1,
    high_percentile: float = 99.9,
) -> InversionParameters:
    """Estimate parameters; explicit dmin/dmax/offset override estimation.

    base_roi should contain ONLY unexposed film, not the illuminated area
    outside the film. range_roi should contain the actual photograph.
    Percentiles are computed independently per RGB channel.
    """
    image = _validate_image(negative_linear)
    if not 0 <= low_percentile < high_percentile <= 100:
        raise ValueError("Require 0 <= low percentile < high percentile <= 100.")
    if dmin is not None and base_roi is not None:
        raise ValueError("Use either dmin or base_roi, not both.")

    # STEP 1: Measure the film base, equivalent to negadoctor's Dmin picker.
    if dmin is not None:
        base = np.asarray(dmin, dtype=np.float64)
    elif base_roi is not None:
        base = _region(image, base_roi).mean(axis=(0, 1))
    else:
        warnings.warn(
            "Dmin is approximated from bright channel percentiles. For reliable "
            "film-base compensation, supply --base-roi x y w h on bare film "
            "or --dmin R G B in linear units.", stacklevel=2,
        )
        base = np.percentile(image, high_percentile, axis=(0, 1))
    _validate_parameters(InversionParameters(tuple(base), 1.0, 0.0))

    # STEP 2: Measure transmission extrema in the photograph region.
    pixels = _region(image, range_roi)
    low = np.maximum(np.percentile(pixels, low_percentile, axis=(0, 1)), THRESHOLD)
    high = np.maximum(np.percentile(pixels, high_percentile, axis=(0, 1)), THRESHOLD)

    # STEP 3: Port apply_auto_Dmax(): darkest transmission gives max density.
    # Difference of logs is equivalent to log10(base / low), avoiding overflow.
    density_max = np.log10(base) - np.log10(low)
    density_range = float(np.max(density_max)) if dmax is None else float(dmax)
    if not np.isfinite(density_range) or density_range <= 1e-12:
        raise ValueError("Cannot estimate a positive density range. Check the "
                         "film-base sample or provide a positive --dmax.")

    # STEP 4: Port apply_auto_offset(): place the low-density end near black.
    density_min = np.log10(base) - np.log10(high)
    bias = float(np.min(density_min / density_range)) if offset is None else float(offset)
    params = InversionParameters(tuple(float(v) for v in base), density_range, bias)
    _validate_parameters(params)
    return params


def invert_negative(
    negative_linear: np.ndarray,
    params: InversionParameters,
) -> np.ndarray:
    """Invert linear RGB with fixed params; return a float32 RGB intermediate.

    Does not modify input. Reuse params for frames with the same film and
    scan settings when consistent rendering across a roll is desired.
    """
    image = _validate_image(negative_linear)
    base = _validate_parameters(params)

    # STEP 5: Normalize by the orange film base and convert to log10 density.
    density = np.log10(base) - np.log10(np.maximum(image, THRESHOLD))

    # STEP 6: Scale density and compensate scanner exposure in log space.
    exponent = -density / params.dmax + params.offset

    # STEP 7: Simulate neutral printing: 1 - 10**exponent reverses brightness.
    # For exponent >= 0 the source clamps the negative result to zero.
    # Clipping the exponent first avoids overflow and has the same result.
    # expm1 improves precision near black compared with subtracting from one.
    positive = -np.expm1(np.minimum(exponent, 0.0) * np.log(10.0))
    return positive.astype(np.float32)


def srgb_to_linear(encoded: np.ndarray) -> np.ndarray:
    """Decode the standard sRGB transfer function; no ICC conversion."""
    x = np.asarray(encoded, dtype=np.float64)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(linear: np.ndarray) -> np.ndarray:
    """Encode a display preview, clipping only at the export boundary."""
    x = np.clip(linear, 0.0, 1.0)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - 0.055)


def load_image(path: Path, encoding: str) -> np.ndarray:
    """Read float .npy, precision-preserving RGB PNG, or 8-bit JPEG."""
    if path.suffix.lower() == ".npy":
        array = np.load(path, allow_pickle=False)
        if array.dtype.kind != "f":
            raise ValueError(".npy input must contain floating-point RGB values.")
    elif path.suffix.lower() == ".png":
        # Pillow can silently reduce 16-bit RGB PNG to 8 bits. OpenCV's
        # unchanged loader preserves uint16; convert BGR to our RGB interface.
        try:
            import cv2
        except ImportError as error:
            raise ValueError("PNG loading requires opencv-python-headless. Run: "
                             "pip install opencv-python-headless") from error
        with path.open("rb") as stream:
            encoded_bytes = np.frombuffer(stream.read(), dtype=np.uint8)
        raw = cv2.imdecode(encoded_bytes, cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise ValueError("Cannot decode PNG input.")
        if raw.ndim != 3 or raw.shape[2] != 3:
            raise ValueError("PNG must have three RGB channels, without alpha.")
        if raw.dtype not in (np.dtype("uint8"), np.dtype("uint16")):
            raise ValueError("PNG must use 8-bit or 16-bit integer channels.")
        scale = np.iinfo(raw.dtype).max  # 255 or 65535, not the image maximum
        array = raw[..., ::-1].astype(np.float64) / scale
    else:
        from PIL import Image
        with Image.open(path) as image:
            if image.mode != "RGB" or image.format != "JPEG":
                raise ValueError("Use RGB PNG, RGB JPEG, or float .npy.")
            array = np.asarray(image, dtype=np.float64) / 255.0
    array = _validate_image(array)
    if encoding == "srgb":
        if np.any(array > 1):
            raise ValueError("sRGB input must be scaled to 0..1.")
        return srgb_to_linear(array)
    if encoding != "linear":
        raise ValueError("encoding must be 'linear' or 'srgb'.")
    return array


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Negadoctor-based film-base compensation and inversion.",
        epilog="Recommended: select bare film with --base-roi, and export .npy "
               "for your teammate's color/exposure correction. Dmin is linear RGB.",
    )
    parser.add_argument("input", type=Path, help="8/16-bit RGB PNG or 8-bit JPEG or float RGB .npy")
    parser.add_argument("output", type=Path, help=".npy intermediate or .png sRGB preview")
    parser.add_argument("--encoding", required=True, choices=["linear", "srgb"],
                        help="Input transfer function; standard PNG usually uses srgb")
    base_group = parser.add_mutually_exclusive_group()
    base_group.add_argument("--dmin", nargs=3, type=float, metavar=("R", "G", "B"),
                            help="Explicit positive film-base RGB in LINEAR units")
    base_group.add_argument("--base-roi", nargs=4, type=int, metavar=("X", "Y", "W", "H"),
                            help="Bare film region; origin is image top-left")
    parser.add_argument("--range-roi", nargs=4, type=int, metavar=("X", "Y", "W", "H"),
                        help="Photograph region for Dmax/offset estimation; exclude borders")
    parser.add_argument("--dmax", type=float, help="Override estimated positive density range")
    parser.add_argument("--offset", type=float, help="Override estimated log-space exposure bias")
    parser.add_argument("--low-percentile", type=float, default=0.1)
    parser.add_argument("--high-percentile", type=float, default=99.9)
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing existing output")
    if argv is None and len(sys.argv) == 1:
        parser.print_help()
        return 0
    args = parser.parse_args(argv)
    try:
        if args.output.suffix.lower() not in {".npy", ".png"}:
            raise ValueError("Output extension must be .npy or .png.")
        if args.output.resolve() == args.input.resolve():
            raise ValueError("Input and output paths must differ.")
        if args.output.exists() and not args.overwrite:
            raise ValueError("Output exists. Use --overwrite to replace it.")
        image = load_image(args.input, args.encoding)
        params = estimate_parameters(
            image, dmin=args.dmin, base_roi=args.base_roi, range_roi=args.range_roi,
            dmax=args.dmax, offset=args.offset, low_percentile=args.low_percentile,
            high_percentile=args.high_percentile,
        )
        positive = invert_negative(image, params)
        if args.output.suffix.lower() == ".npy":
            np.save(args.output, positive, allow_pickle=False)
        else:
            from PIL import Image
            preview = np.round(linear_to_srgb(positive) * 255).astype(np.uint8)
            Image.fromarray(preview).save(args.output)
            print("PNG is an 8-bit sRGB preview. Use .npy for subsequent correction.")
        print("Parameters:", asdict(params))
        print("Saved:", args.output)
        print("Next step: correct_color_exposure(np.load(output_path)). "
              "Input to that stage is float32, linear RGB, H x W x 3.")
        return 0
    except (ValueError, OSError) as error:
        parser.error(f"{error} See --help for input and sampling instructions.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

