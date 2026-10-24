"""Automatic color and exposure correction for inverted negatives."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from negative_inversion import linear_to_srgb, load_image


def stretch_channels(image, low=0.5, high=99.5):
    black = np.percentile(image, low, axis=(0, 1))
    white = np.percentile(image, high, axis=(0, 1))
    return np.clip((image - black) / np.maximum(white - black, 1e-6), 0, 1)


def gray_world(image):
    means = np.maximum(image.reshape(-1, 3).mean(axis=0), 1e-6)
    return np.clip(image * means.mean() / means, 0, 1)


def auto_exposure(image, low=0.18, high=0.5, black=0.01, max_gamma=2.0):
    luminance = image @ np.array([0.2126, 0.7152, 0.0722])
    lit = luminance[luminance > black]
    median = np.clip(np.median(lit if lit.size else luminance), 1e-4, 0.9999)
    gamma = np.log(np.clip(median, low, high)) / np.log(median)
    return image ** np.clip(gamma, 1 / max_gamma, max_gamma)


def boost_saturation(srgb, saturation=1.2):
    lab = cv2.cvtColor(srgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[..., 1:] = np.clip((lab[..., 1:] - 128) * saturation + 128, 0, 255)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)


def correct_color_exposure(positive_linear, saturation=1.2):
    image = stretch_channels(positive_linear)
    image = gray_world(image)
    image = auto_exposure(image)
    srgb = np.round(linear_to_srgb(image) * 255).astype(np.uint8)
    return boost_saturation(srgb, saturation)


def save_rgb(path, rgb):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"Could not save {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Inverted .npy (linear) or .png (sRGB)")
    parser.add_argument("output", type=Path, help="Corrected sRGB .png")
    parser.add_argument("--saturation", type=float, default=1.2)
    args = parser.parse_args()

    encoding = "linear" if args.input.suffix.lower() == ".npy" else "srgb"
    positive = load_image(args.input, encoding)
    save_rgb(args.output, correct_color_exposure(positive, args.saturation))
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
