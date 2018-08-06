import os
import cv2
import argparse
from matplotlib import cm
from matplotlib import pyplot as plt
from doc_scanner import scanner

parser = argparse.ArgumentParser()
parser.add_argument("--show", dest='show', default='mpl')
parser.add_argument("--image-path", dest='image_path', default='./data/images')
options = parser.parse_args()

files = os.listdir(options.image_path)
for file in files:
    filepath = os.path.join(options.image_path, file)
    if os.path.isdir(filepath):
        continue
    else:
        if not filepath.endswith('jpg'):
            continue
    image = cv2.imread(os.path.join(options.image_path, file))

    # resize image
    height, width, _ = image.shape
    if height > width:
        resize_ratio = width / 500
    else:
        resize_ratio = height / 500
    resized_image = cv2.resize(image, (0, 0), fx=1 / resize_ratio, fy=1 / resize_ratio, interpolation=cv2.INTER_AREA)

    # Convert RGB to HSV colorspace
    hsv = cv2.cvtColor(resized_image, cv2.COLOR_RGB2HSV)

    # hue ranges from 0-180
    # hue = scanner(hsv[:, :, 0])
    intensity = scanner(hsv[:, :, 2])
    intensity.scan()
    if intensity.corners is not None:
        warped = intensity.warp(image, scale=resize_ratio)
    else:
        saturation = scanner(hsv[:, :, 1])
        saturation.scan()
        if saturation.corners is not None:
            warped = saturation.warp(image, scale=resize_ratio)
        else:
            warped = None

    # plt.clf()
    # plt.ion()
    #
    # ax = plt.subplot(2, 1, 1)
    # ax.imshow(image)
    #
    # ax = plt.subplot(2, 1, 2)
    # ax.imshow(warped)
    #
    # plt.pause(0.2)
    # plt.waitforbuttonpress()
    cv2.imshow("Original", image)
    cv2.imshow("Warped", warped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
