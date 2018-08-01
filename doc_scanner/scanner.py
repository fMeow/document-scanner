import cv2
import numpy as np
from dataclasses import dataclass
import skimage.transform


@dataclass
class HoughResult:
    h: np.array
    theta: np.array
    distance: np.array


@dataclass
class ProcessingResult:
    """
    blurred ,edges ,contour_image ,hist_equalized are images of the same shape of input image.
    hist is a 1-D array of histogram.
    contours is opencv result of all contours
    hough is the result of line hough transformation in image-scikit
    """
    blurred: np.array
    edges: np.array
    contour_image: np.array
    hist_equalized: np.array

    hist: np.array

    contours: np.array

    hough: HoughResult


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
    _, contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_image = cv2.drawContours(np.zeros(image.shape[0:2]), contours, -1, (128, 255, 0), 3)
    theta = np.linspace(-np.pi * 1 / 4, np.pi * 3 / 4, 180)
    hough = skimage.transform.hough_line(contour_image, theta)
    return ProcessingResult(blurred=blurred, edges=edges, contour_image=contour_image, hist_equalized=hist_equalized,
                            hist=hist, hough=HoughResult(*hough), contours=contours)
