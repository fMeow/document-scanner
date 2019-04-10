import os
import pathlib
import argparse

import cv2
import numpy as np
from PIL import Image

from doc_scanner import scanner

def scan(path, ):
    image = cv2.imread(path)

    # ----------------------------------------
    # Reshape and scan
    # ----------------------------------------

    height, width, _ = image.shape
    if height > width:
        resize_ratio = width / 500
    else:
        resize_ratio = height / 500
    resized_image = cv2.resize(image, (0, 0), interpolation=cv2.INTER_AREA,
                               fx=1 / resize_ratio, fy=1 / resize_ratio, )

    # Convert RGB to HSV colorspace
    hsv = cv2.cvtColor(resized_image, cv2.COLOR_RGB2HSV)

    """
    First scan gray scale component of images and if failed, then turn to saturation images.
    """
    # hue ranges from 0-180
    # hue = scanner(hsv[:, :, 0])
    intensity = scanner(hsv[:, :, 2])
    intensity.scan()
    if intensity.corners is not None:
        # Find corners in intensity images
        warped = (True, intensity.warp(image, scale=resize_ratio))
    else:
        saturation = scanner(hsv[:, :, 1])
        saturation.scan()
        if saturation.corners is not None:
            warped = (True, saturation.warp(image, scale=resize_ratio))
        else:
            warped = (False, image)
    scan_ok, warped_image = warped

    # ----------------------------------------
    # Reshape and rotate
    # ----------------------------------------

    # normal document
    if height < width:
        im = Image.fromarray(warped_image).rotate(-90, expand=1)
        warped_image = np.array(im)

    if width > 1280:
        resize_ratio = width / 1280
        warped_image = cv2.resize(warped_image, (0, 0), interpolation=cv2.INTER_AREA, fx=1 / resize_ratio,
                                    fy=1 / resize_ratio, )

    # Only for non ID card document
    result = cv2.GaussianBlur(warped_image, (3, 3), 5)
    return scan_ok, result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--from_dir", dest='from_dir', default='./data')
    parser.add_argument("--to_dir", dest='to_dir', default='./output')
    args = parser.parse_args()

    success_dir = os.path.join(args.to_dir, 'success')
    fail_dir = os.path.join(args.to_dir, 'fail')

    pathlib.Path(success_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(fail_dir).mkdir(parents=True, exist_ok=True)

    files = os.listdir(args.from_dir)
    for i, file in enumerate(files):
        filepath = os.path.join(args.from_dir, file)
        if os.path.isdir(filepath):
            continue
        else:
            if not filepath.endswith('jpg'):
                continue
        ok, result = scan(filepath)

        if ok:
            path = os.path.join(success_dir, file)
        else:
            path = os.path.join(fail_dir, file)
        cv2.imwrite(path, result)
        print(f"{i}/{len(files)}", end='\r', flush=True)
