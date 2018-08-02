import cv2
import numpy as np
import pandas as pd
import itertools
from dataclasses import dataclass
from skimage.transform import hough_line_peaks, hough_line


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
    hough = hough_line(contour_image, theta)
    return ProcessingResult(blurred=blurred, edges=edges, contour_image=contour_image, hist_equalized=hist_equalized,
                            hist=hist, hough=HoughResult(*hough), contours=contours)


def select_edge(result: ProcessingResult, ax=None, image: np.array = None):
    lines = hough_line_peaks(result.hough.h, result.hough.theta, result.hough.distance, min_distance=10, min_angle=50,
                             threshold=0.5 * result.hough.h.max(), num_peaks=np.inf)
    lines = pd.DataFrame(np.array(lines).T, columns=['hits', 'angle', 'intercept'])
    _divide_line_orientation(lines)
    intersections = _find_intersections(lines, result.contour_image)

    if ax and image is not None:
        for ix, line in lines.iterrows():
            x = (0, image.shape[1])
            y = find_point_polar(line, x)
            if line['direction'] == 'vertical':
                color = 'r'
            elif line['direction'] == 'horizontal':
                color = 'g'
            else:
                color = 'k'
            ax.plot(x, y, '-{}'.format(color))

        try:
            x = intersections['x'].values
            y = intersections['y'].values
            ax.plot(x, y, 'bx', ms=20)
        except KeyError:
            pass
    return list(lines)


def _divide_line_orientation(lines: pd.DataFrame, err=np.pi * 1 / 12, inplace: bool = True):
    """
    Discriminate between horizontal and vertical lines
    :param lines: lines in polar coordination
    :param err:
    :param inplace:
    :return:
    """
    if not inplace:
        out = lines.copy()
    else:
        out = lines
    for ix, line in out.iterrows():
        if abs(line['angle']) < err:
            # vertical
            direction = 'vertical'
        elif abs(line['angle'] - np.pi / 2) < err:
            # horizontal
            direction = 'horizontal'
        else:
            # irrelevant lines
            direction = 'irrelevant'
        out._set_value(ix, 'direction', direction)
    return out


def find_point_polar(line: pd.DataFrame, x: tuple):
    angle = line['angle']
    dist = line['intercept']
    y = tuple(map(lambda i: (dist - i * np.cos(angle)) / np.sin(angle), x))
    return y


def _pairwise_intersection(horizontal, vertical, contour_image=None, along_length=50):
    def line(p1, p2):
        """
        compute Ax+By=C given point (x1,y1) and (x2,y2)
        :param p1:
        :param p2:
        :return:
        """
        a = (p1[1] - p2[1])
        b = (p2[0] - p1[0])
        c = (p1[0] * p2[1] - p2[0] * p1[1])
        return a, b, -c

    def intersection(L1, L2):
        """
        Compute intersection given two lines.
        L=(A,B,C) while Ax+By=C
        :param L1:
        :param L2:
        :return:
        """
        d = L1[0] * L2[1] - L1[1] * L2[0]
        dx = L1[2] * L2[1] - L1[1] * L2[2]
        dy = L1[0] * L2[2] - L1[2] * L2[0]
        if d != 0:
            x = dx / d
            y = dy / d
            return x, y
        else:
            raise Exception("Lines given are parallel")

    if contour_image is not None:
        x = (0, contour_image.shape[1])
    else:
        x = (0, 1000)
    y_h = find_point_polar(horizontal, x)
    y_v = find_point_polar(vertical, x)

    point = pd.DataFrame([intersection(line(*tuple(zip(x, y_h))), line(*tuple(zip(x, y_v))))], columns=['x', 'y'])

    return point


def _find_intersections(lines, contour_image=None):
    try:
        lines['direction']
    except ValueError:
        _divide_line_orientation(lines)
    vertical_lines = lines[lines['direction'] == 'vertical']
    horizontal_lines = lines[lines['direction'] == 'horizontal']
    combinations = itertools.product(vertical_lines.iterrows(), horizontal_lines.iterrows())
    intersections = pd.DataFrame()
    for (_, vertical_line), (_, horizontal_line) in combinations:
        point = _pairwise_intersection(horizontal_line, vertical_line, contour_image)
        intersections = intersections.append(point)
    return intersections

# class scanner:
#     def __init__(self, image):
#         """
#         :param image: RGB
#         """
#         self.__image = image
#
#         # Convert RGB to HSV colorspace
#         hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
#
#         # hue ranges from 0-180
#         self.hue = hsv[:, :, 0]
#         self.saturation = hsv[:, :, 1]
#         self.intensity = hsv[:, :, 2]
#
#     def __preprocess(self, img, blur_size=25, morphology_kernel_size=15, intensity_lower=0, intensity_upper=255):
#         """Preprocess pipeline
#         1. Blur
#         1. Histogram equalization
#         1. Morphological operation (Opening)
#         1. (Optional) Threshold based segmentation.
#
#             Here we assume that the document of interest is mainly white while background is darker.
#             Then we can extract document from background with a proper threshold.
#             After histogram, maybe we can just assume the document lays in the half brighter part on histogram.
#         :param img: 2D image (Hue or saturation or intensity)
#         :param blur_size:
#         :param morphology_kernel_size:
#         :param intensity_lower:
#         :param intensity_upper:
#         :return:
#         """
#         # blurred = cv2.GaussianBlur(image, (5, 5), 0)
#         # blurred = cv2.bilateralFilter(image, 9, 50, 50)
#         blurred = cv2.medianBlur(img, blur_size)
#         hist_equalized = cv2.equalizeHist(blurred)
#
#         # Morphological Open operation
#         # Determine kernel size according to a priori knowledge on the size of words
#         kernel = np.ones((morphology_kernel_size, morphology_kernel_size), dtype=np.int8)
#         hist_equalized = cv2.morphologyEx(hist_equalized, cv2.MORPH_OPEN, kernel)
#         hist_equalized = cv2.morphologyEx(hist_equalized, cv2.MORPH_CLOSE, kernel)
#
#         # hist = cv2.calcHist([hist_equalized], [0], None, [256], [0, 256])
#         # plt.bar(np.arange(len(hist)), hist.flatten())
#         # plt.show()
#
#         # TODO intensity threshold filter can bring artifacts
#         # Threshold the intensity image or gray scale image
#         # ret, thresh = cv2.threshold(imgray, 127, 255, 0)
#         mask = cv2.inRange(hist_equalized, intensity_lower, intensity_upper)
#
#         # Bitwise-AND mask and original image
#         filtered = cv2.bitwise_and(hist_equalized, hist_equalized, mask=mask)
#         return filtered
