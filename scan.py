"""
Basic idea:
1. Convert color space from RGB to HSV
"""

import os
import cv2
import argparse
import numpy as np
from skimage.transform import hough_line_peaks
from matplotlib import cm
from matplotlib import pyplot as plt
from doc_scanner.scanner import filter_and_edge_detect, edge_selection

parser = argparse.ArgumentParser()
parser.add_argument("--show", dest='show', default='mpl')
parser.add_argument("--image-path", dest='image_path', default='./images')
options = parser.parse_args()

files = os.listdir(options.image_path)
for file in files:
    if os.path.isdir(os.path.join(options.image_path, file)):
        continue
    image = cv2.imread(os.path.join(options.image_path, file))

    # resize image
    height, width, _ = image.shape
    if height > width:
        resize_ratio = width / 500
    else:
        resize_ratio = height / 500
    image = cv2.resize(image, (0, 0), fx=1 / resize_ratio, fy=1 / resize_ratio, interpolation=cv2.INTER_AREA)

    # Convert RGB to HSV colorspace
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # hue ranges from 0-180
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    intensity = hsv[:, :, 2]

    hue_result = filter_and_edge_detect(hue, canny_lower=30, canny_upper=150)
    saturation_result = filter_and_edge_detect(saturation)
    intensity_result = filter_and_edge_detect(intensity)

    fused = (intensity_result.edges + saturation_result.edges) / 2
    fused = fused.astype(np.uint8)

    if options.show == 'mpl':
        plt.clf()
        plt.ion()
        ax = plt.subplot(3, 3, 1)
        ax.imshow(image)
        ax.set_title("Original")

        # ax = plt.subplot(2, 3, 2)
        # ax.imshow(hue_edges, cmap='gray')
        # ax.set_title("Hue")
        ax = plt.subplot(3, 3, 2)
        ax.imshow(intensity_result.contour_image, cmap='gray')
        ax.set_title("Contours(Intensity)")

        ax = plt.subplot(3, 3, 3)
        ax.imshow(saturation_result.contour_image, cmap='gray')
        ax.set_title("Saturation")

        ax = plt.subplot(3, 3, 4)
        ax.bar(np.arange(0, 256), intensity_result.hist.flatten())
        ax.set_title("Intensity Histogram")

        ax = plt.subplot(3, 3, 5)
        # ax.imshow(intensity_with_contours)
        # ax.set_title("Contours(Intensity)")
        ax.imshow(np.log(1 + intensity_result.hough.h), cmap=cm.gray, aspect=1 / 1.5,
                  extent=[np.rad2deg(intensity_result.hough.theta[-1]), np.rad2deg(intensity_result.hough.theta[0]),
                          500, 300],
                  )
        ax.set_title('Hough transform(Intensity)')
        # ax.set_xlabel('Angles (degrees)')
        ax.set_ylabel('Distance (pixels)')
        ax.axis('image')

        ax = plt.subplot(3, 3, 6)
        ax.imshow(np.log(1 + saturation_result.hough.h), cmap=cm.gray, aspect=1 / 1.5,
                  extent=[np.rad2deg(saturation_result.hough.theta[-1]), np.rad2deg(saturation_result.hough.theta[0]),
                          500, 300],
                  )
        ax.set_title('Hough transform(Saturation)')
        # ax.set_xlabel('Angles (degrees)')
        ax.set_ylabel('Distance (pixels)')
        ax.axis('image')

        ax = plt.subplot(3, 3, 8)
        ax.imshow(image)
        edge_selection(intensity_result, ax, image)
        ax.set_xlim((0, image.shape[1]))
        ax.set_ylim((image.shape[0], 0))
        ax.set_axis_off()
        ax.set_title('Detected lines(Intensity)')

        ax = plt.subplot(3, 3, 9)
        ax.imshow(image)
        edge_selection(saturation_result, ax, image)
        ax.set_xlim((0, image.shape[1]))
        ax.set_ylim((image.shape[0], 0))
        ax.set_axis_off()
        ax.set_title('Detected lines(Saturation)')

        # plt.tight_layout()
        plt.pause(0.2)
        plt.waitforbuttonpress()

    elif options.show == 'cv':
        # cv2.imshow('Gray Scale', gray)
        # cv2.imshow('opening', hist_equalized)
        # cv2.imshow('Intensity Filtered', filtered)

        cv2.imshow('saturation edge', saturation_result.edges)
        cv2.imshow('intensity edge', intensity_result.edges)
        # cv2.imshow('hue edge', hue_edges)
        cv2.imshow('original', image)
        # cv2.imshow('intensity', intensity_blurred)
        # cv2.imshow('intensity contour', intensity_with_contours)
        # cv2.imshow('saturation contour', saturation_with_contours)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        raise ValueError("--show must be cv or mpl")
