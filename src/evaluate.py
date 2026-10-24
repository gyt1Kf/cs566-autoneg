"""Compare inversion and color correction against reference images."""

import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
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
    return psnr, ssim


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

        psnr_before, ssim_before = scores(before, reference)
        psnr_after, ssim_after = scores(after, reference)
        rolls[name.split("-")[0]].append((psnr_before, ssim_before, psnr_after, ssim_after))
        print(f"{name}: before PSNR {psnr_before:.2f} SSIM {ssim_before:.3f} | "
              f"after PSNR {psnr_after:.2f} SSIM {ssim_after:.3f}")

        save_rgb(args.output / f"{name}.corrected.png", after)

    print()
    for roll, values in sorted(rolls.items()):
        mean = np.mean(values, axis=0)
        print(f"{roll} ({len(values)} images): before PSNR {mean[0]:.2f} SSIM {mean[1]:.3f} | "
              f"after PSNR {mean[2]:.2f} SSIM {mean[3]:.3f}")


if __name__ == "__main__":
    main()
