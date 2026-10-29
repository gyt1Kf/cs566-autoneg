
"""Color negative restoration pipeline."""

import argparse
from pathlib import Path
import numpy as np

from negative_inversion import (
    estimate_parameters,
    invert_negative,
    linear_to_srgb,
    load_image,
)
from color_correction import correct_color_exposure, save_rgb


def run_pipeline(
    input_path,
    output_path,
    encoding="linear",
    base_roi=None,
    range_roi=None,
    saturation=1.2,
):
    """Invert a color negative and automatically correct its colors."""

    input_path = Path(input_path)
    output_path = Path(output_path)
    negative = load_image(input_path, encoding)

    params = estimate_parameters(
        negative,
        base_roi=base_roi,
        range_roi=range_roi,
    )

    positive_linear = invert_negative(negative, params)

    corrected_rgb = correct_color_exposure(
        positive_linear,
        saturation=saturation,
    )

    save_rgb(output_path, corrected_rgb)

    intermediate_path = output_path.with_name(
        output_path.stem + "_inverted.npy"
    )
    np.save(intermediate_path, positive_linear, allow_pickle=False)

    print("Pipelined completed successfully.")
    print("Inversion parameters:", params)
    print("Final image:", output_path)
    print("Linear intermediate:", intermediate_path)

def main():
    parser = argparse.ArgumentParser(
        description="Inverts and color corrects a film negative automatically."
    )

    parser.add_argument("input", type=Path, help="Scanned negative image")
    parser.add_argument("output", type=Path, help="Output corrected PNG")
    parser.add_argument(
        "--encoding",
        choices=["srgb", "linear"],
        default="linear",
        help="Input transfer function (default: linear, matches dng_to_png.py output)",
    )
    parser.add_argument(
        "--base-roi",
        nargs=4,
        type=int,
        metavar=("X", "Y", "W", "H"),
        help="Bare film region for film-base estimation",
    )
    parser.add_argument(
        "--range-roi",
        nargs=4,
        type=int,
        metavar=("X", "Y", "W", "H"),
        help="Photograph region for density estimation",
    )
    parser.add_argument(
        "--saturation",
        type=float,
        default=1.2,
        help="Saturation multiplier (default: 1.2)",
    )

    args = parser.parse_args()

    run_pipeline(
        args.input,
        args.output,
        encoding=args.encoding,
        base_roi=tuple(args.base_roi) if args.base_roi else None,
        range_roi=tuple(args.range_roi) if args.range_roi else None,
        saturation=args.saturation,
    )

if __name__ == "__main__":
    main()
