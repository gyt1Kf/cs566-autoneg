"""Compare inversion and color correction against reference images."""

import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from skimage.color import deltaE_ciede2000, rgb2lab
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from color_correction import correct_color_exposure, save_rgb
from negative_inversion import load_image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BLUENEG = PROJECT_ROOT / "data" / "blueneg"


def read_rgb(path):
    return cv2.cvtColor(cv2.imread(str(path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def find_homography(image, reference, min_inliers=15):
    orb = cv2.ORB_create(5000)
    gray_image = cv2.equalizeHist(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))
    gray_reference = cv2.equalizeHist(cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY))
    keys_image, desc_image = orb.detectAndCompute(gray_image, None)
    keys_reference, desc_reference = orb.detectAndCompute(gray_reference, None)
    if desc_image is None or desc_reference is None:
        return None
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(desc_image, desc_reference)
    if len(matches) < min_inliers:
        return None
    source = np.float32([keys_image[m.queryIdx].pt for m in matches])
    target = np.float32([keys_reference[m.trainIdx].pt for m in matches])
    homography, inliers = cv2.findHomography(source, target, cv2.RANSAC, 3.0)
    if homography is None or inliers.sum() < min_inliers:
        return None
    return homography


def align(image, reference, homography):
    size = (reference.shape[1], reference.shape[0])
    if homography is None:
        return cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    return cv2.warpPerspective(image, homography, size, flags=cv2.INTER_AREA)


def scores(image, reference):
    psnr = peak_signal_noise_ratio(reference, image)
    ssim = structural_similarity(reference, image, channel_axis=2)
    lab_image, lab_reference = rgb2lab(image), rgb2lab(reference)
    delta_e = deltaE_ciede2000(lab_reference, lab_image).mean()
    # Chroma-only error ignores lightness, isolating color cast from tone curve.
    chroma = np.linalg.norm(lab_image[..., 1:] - lab_reference[..., 1:], axis=2).mean()
    return psnr, ssim, delta_e, chroma


def format_scores(values):
    psnr, ssim, delta_e, chroma = values
    return f"PSNR {psnr:.2f} SSIM {ssim:.3f} dE00 {delta_e:.2f} dab {chroma:.2f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=BLUENEG / "negative-preview-8bit" / "blue-corrupted")
    parser.add_argument("--reference", type=Path, default=BLUENEG / "pseudogt-8bit" / "blue-corrupted")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results")
    args = parser.parse_args()

    rolls = defaultdict(list)
    for reference_path in sorted(args.reference.glob("*.png")):
        name = reference_path.name.split(".")[0]
        candidates = sorted(args.input.glob(f"{name}.*png"))
        if not candidates:
            print(f"Skipped {name}: no input image")
            continue
        input_path = candidates[0]

        reference = read_rgb(reference_path)
        before = read_rgb(input_path)
        after = correct_color_exposure(load_image(input_path, "srgb"))

        homography = find_homography(before, reference)
        before = align(before, reference, homography)
        after = align(after, reference, homography)

        score_before = scores(before, reference)
        score_after = scores(after, reference)
        rolls[name.split("-")[0]].append(score_before + score_after)
        print(f"{name}: before {format_scores(score_before)} | after {format_scores(score_after)}")

        save_rgb(args.output / f"{name}.corrected.png", after)

    print()
    for roll, values in sorted(rolls.items()):
        mean = np.mean(values, axis=0)
        print(f"{roll} ({len(values)} images): before {format_scores(mean[:4])} | "
              f"after {format_scores(mean[4:])}")


if __name__ == "__main__":
    main()
