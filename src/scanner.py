import cv2
import os
import numpy as np
from matplotlib import pyplot as plt


def filter_and_edge_detect(image, kernel_size=15, intensity_lower=0, intensity_upper=255, canny_lower=10,
                           canny_upper=70):
    """Filter and edge detection given a 2D digital image array
    1. Blur
    1. Histogram equalization
    1. Morphological operation (Opening)
    1. (Optional) Threshold based segmentation.

        Here we assume that the document of interest is mainly white while background is darker.
        Then we can extract document from background with a proper threshold.
        After histogram, maybe we can just assume the document lays in the half brighter part on histogram.
    1. Canny edge detector

    :param image:
    :param kernel_size:
    :param intensity_lower:
    :param intensity_upper:
    :param canny_lower:
    :param canny_upper:
    :return:
    """
    # blurred = cv2.GaussianBlur(image, (5, 5), 0)
    # blurred = cv2.bilateralFilter(image, 9, 50, 50)
    blurred = cv2.medianBlur(image, 25)
    hist_equalized = cv2.equalizeHist(blurred)

    # Morphological Open operation
    # Determine kernel size according to a priori knowledge on the size of words
    kernel = np.ones((kernel_size, kernel_size), dtype=np.int8)
    hist_equalized = cv2.morphologyEx(hist_equalized, cv2.MORPH_OPEN, kernel)
    hist_equalized = cv2.morphologyEx(hist_equalized, cv2.MORPH_CLOSE, kernel)

    hist = cv2.calcHist([hist_equalized], [0], None, [256], [0, 256])
    # plt.bar(np.arange(len(hist)), hist.flatten())
    # plt.show()

    # TODO intensity threshold filter can bring artifacts
    # Threshold the intensity image or gray scale image
    # ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    mask = cv2.inRange(hist_equalized, intensity_lower, intensity_upper)

    # Bitwise-AND mask and original image
    filtered = cv2.bitwise_and(hist_equalized, hist_equalized, mask=mask)

    # TODO decide thresholds
    edges = cv2.Canny(filtered, canny_lower, canny_upper, L2gradient=True, apertureSize=3)
    return edges, (hist_equalized, hist), blurred


if __name__ == '__main__':
    """
    Basic idea:
    1. Convert color space from RGB to HSV
    """
    files = os.listdir('../images')
    for file in files:
        if os.path.isdir(os.path.join('../images', file)):
            continue
        image = cv2.imread(os.path.join('../images', file))

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

        hue_edges, (_, hue_hist), hue_blurred = filter_and_edge_detect(hue, canny_lower=30, canny_upper=150)
        saturation_edges, (_, saturation_hist), saturation_blurred = filter_and_edge_detect(saturation)
        intensity_edges, (_, intensity_hist), intensity_blurred = filter_and_edge_detect(intensity)

        fused = (intensity_edges + saturation_edges) / 2
        fused = fused.astype(np.uint8)

        # cv2.HoughLines(edges,)

        _, contours, _ = cv2.findContours(intensity_edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # contours = sorted(contours, key=cv2.contourArea, reverse=True)
        intensity_with_contours = cv2.drawContours(image.copy(), contours, -1, (128, 255, 0), 3)
        lines = cv2.HoughLines(intensity_edges, 1, np.pi / 180, 100)
        if lines is not None:
            for line in lines:
                for rho, theta in line:
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a * rho
                    y0 = b * rho
                    x1 = int(x0 + 1000 * (-b))
                    y1 = int(y0 + 1000 * (a))
                    x2 = int(x0 - 1000 * (-b))
                    y2 = int(y0 - 1000 * (a))
                    cv2.line(intensity_with_contours, (x1, y1), (x2, y2), (0, 0, 255), 2)

        _, contours_saturation, _ = cv2.findContours(saturation_edges.copy(), cv2.RETR_EXTERNAL,
                                                     cv2.CHAIN_APPROX_SIMPLE)
        # contours = sorted(contours, key=cv2.contourArea, reverse=True)
        saturation_with_contours = cv2.drawContours(image.copy(), contours_saturation, -1, (128, 255, 0), 3)
        # threshold of hit points
        # may be more appropriate with scikit-image
        # opencv hough transform miss wanted information about hit numbers
        lines = cv2.HoughLines(saturation_edges, 1, np.pi / 180, 100)
        if lines is not None:
            for line in lines:
                for rho, theta in line:
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a * rho
                    y0 = b * rho
                    x1 = int(x0 + 1000 * (-b))
                    y1 = int(y0 + 1000 * (a))
                    x2 = int(x0 - 1000 * (-b))
                    y2 = int(y0 - 1000 * (a))
                    cv2.line(saturation_with_contours, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # plt.clf()
        # plt.ion()
        # ax = plt.subplot(2, 3, 1)
        # ax.imshow(image)
        # ax.set_title("Original")
        #
        # ax = plt.subplot(2, 3, 2)
        # ax.imshow(hue_edges, cmap='gray')
        # ax.set_title("Hue")
        #
        # ax = plt.subplot(2, 3, 3)
        # ax.imshow(intensity_edges, cmap='gray')
        # ax.set_title("Intensity")
        #
        # ax = plt.subplot(2, 3, 4)
        # ax.imshow(saturation_edges, cmap='gray')
        # ax.set_title("Saturation")
        #
        # ax = plt.subplot(2, 3, 5)
        # ax.bar(np.arange(0, 256), intensity_hist.flatten())
        # ax.set_title("Intensity Histogram")
        #
        # ax = plt.subplot(2, 3, 6)
        # ax.imshow(image_with_contours)
        # ax.set_title("Image with contours")
        # # plt.tight_layout()
        #
        # plt.pause(0.2)
        # plt.waitforbuttonpress()

        # cv2.imshow('Gray Scale', gray)
        # cv2.imshow('opening', hist_equalized)
        # cv2.imshow('Intensity Filtered', filtered)

        cv2.imshow('saturation edge', saturation_edges)
        cv2.imshow('intensity edge', intensity_edges)
        # cv2.imshow('hue edge', hue_edges)
        cv2.imshow('original', image)
        # cv2.imshow('intensity', intensity_blurred)
        cv2.imshow('intensity contour', intensity_with_contours)
        cv2.imshow('saturation contour', saturation_with_contours)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
