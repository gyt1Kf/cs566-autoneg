"""Convert DNG scans to resized 16-bit PNGs without negative inversion."""

import argparse
from pathlib import Path

import cv2
import rawpy


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def convert_dng(source: Path, destination: Path, max_side: int):
    with rawpy.imread(str(source)) as raw:
        # Keep processing consistent across images.
        # Preserve negative polarity; do not apply automatic brightness.
        rgb = raw.postprocess(
            output_bps=16,
            gamma=(1, 1),
            no_auto_bright=True,
            use_camera_wb=False,
            use_auto_wb=False,
            user_wb=[1.0, 1.0, 1.0, 1.0],
        )

    height, width = rgb.shape[:2]

    if max(height, width) > max_side:
        scale = max_side / max(height, width)
        new_width = max(1, round(width * scale))
        new_height = max(1, round(height * scale))
        rgb = cv2.resize(
            rgb,
            (new_width, new_height),
            interpolation=cv2.INTER_AREA,
        )

    destination.parent.mkdir(parents=True, exist_ok=True)

    # rawpy returns RGB; OpenCV writes images in BGR channel order.
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(destination), bgr):
        raise RuntimeError(f"Could not save {destination}")

    out_height, out_width = rgb.shape[:2]
    print(
        f"{source.name}: {width}x{height} -> "
        f"{out_width}x{out_height}, 16-bit PNG"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data" / "blueneg" / "negative-16bit",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "samples" / "negative",
    )
    parser.add_argument("--max-side", type=int, default=2000)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing PNG files.",
    )
    args = parser.parse_args()

    if args.max_side < 1:
        parser.error("--max-side must be positive")
    if not args.input.is_dir():
        parser.error(f"Input directory does not exist: {args.input}")

    files = sorted(
        path
        for path in args.input.rglob("*")
        if path.is_file() and path.suffix.lower() == ".dng"
    )

    if not files:
        print(f"No DNG files found in {args.input}")
        return

    failed = 0
    for source in files:
        relative = source.relative_to(args.input).with_suffix(".png")
        destination = args.output / relative

        if destination.exists() and not args.overwrite:
            print(f"Skipped existing file: {destination.name}")
            continue

        try:
            convert_dng(source, destination, args.max_side)
        except Exception as error:
            failed += 1
            print(f"FAILED {source.name}: {error}")

    print(f"Finished. Found {len(files)} DNG files; failures: {failed}.")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()