
"""End-to-end color negative restoration pipeline."""

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
    encoding="srgb",
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

    print("Pipeline complete!")
    print("Inversion parameters:", params)
    print("Final image:", output_path)
    print("Linear intermediate:", intermediate_path)


if __name__ == "__main__":
    main()
