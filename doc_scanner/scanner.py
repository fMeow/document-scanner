import cv2
import numpy as np
import pandas as pd
import itertools
from dataclasses import dataclass
from skimage.transform import hough_line_peaks, hough_line
from doc_scanner.math_utils import points2line, find_point_polar, intersection, interpolate_pixels_along_line
from doc_scanner.transform import four_point_transform
from doc_scanner.intersection import calc_intersections, connectivity


class scanner:
    def __init__(self, image: np.array):
        if len(image.shape) != 2:
            raise ValueError("Image should be a single channel 2D array")
        self.image = image

    def scan(self):
        self.preprocess()
        self.hough_transform()
        self.intersections()
        self.connectivity()
        self.corner_detection()
        self.warp()
        return

    def preprocess(self, kernel_size=15, intensity_lower=0, intensity_upper=255, canny_lower=10, canny_upper=70):
        """Filter and edge detection given a 2D digital image array
        1. Blur
        1. Histogram equalization
        1. Morphological operation (Opening)
        1. (Optional) Threshold based segmentation.

            Here we assume that the document of interest is mainly white while background is darker.
            Then we can extract document from background with a proper threshold.
            After histogram, maybe we can just assume the document lays in the half brighter part on histogram.
        1. (Canny) edge detector

        :param kernel_size:
        :param intensity_lower:
        :param intensity_upper:
        :param canny_lower:
        :param canny_upper:
        :return:
        """
        # self.blurred = cv2.GaussianBlur(image, (5, 5), 0)
        # self.blurred = cv2.bilateralFilter(image, 9, 50, 50)
        self.blurred = cv2.medianBlur(self.image, 25)
        self.hist_equalized = cv2.equalizeHist(self.blurred)

        # Morphological Open operation
        # Determine kernel size according to a priori knowledge on the size of words
        kernel = np.ones((kernel_size, kernel_size), dtype=np.int8)
        self.hist_equalized = cv2.morphologyEx(self.hist_equalized, cv2.MORPH_OPEN, kernel)
        self.hist_equalized = cv2.morphologyEx(self.hist_equalized, cv2.MORPH_CLOSE, kernel)

        # hist = cv2.calcHist([self.hist_equalized], [0], None, [256], [0, 256])
        # plt.bar(np.arange(len(hist)), hist.flatten())
        # plt.show()

        # TODO intensity threshold filter can bring artifacts
        # Threshold the intensity image or gray scale image
        # ret, thresh = cv2.threshold(imgray, 127, 255, 0)
        mask = cv2.inRange(self.hist_equalized, intensity_lower, intensity_upper)

        # Bitwise-AND mask and original image
        self.filterred = cv2.bitwise_and(self.hist_equalized, self.hist_equalized, mask=mask)

        # TODO decide canny thresholds
        # TODO use hough line transform instead of canny edge detector
        edges = cv2.Canny(self.filterred, canny_lower, canny_upper, L2gradient=True, apertureSize=3)
        _, contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.edges_img = cv2.drawContours(np.zeros(self.image.shape[0:2]), contours, -1, (128, 255, 0), 3)

    def hough_transform(self):
        theta = np.linspace(-np.pi * 1 / 4, np.pi * 3 / 4, 180)
        h, theta, distance = hough_line(self.edges_img, theta)
        lines = hough_line_peaks(h, theta, distance, min_distance=10, min_angle=50, threshold=0.49 * h.max(),
                                 num_peaks=np.inf)
        # hits, phi, rho = hough_line_peaks(h, theta, distance, min_distance=10, min_angle=50, threshold=0.49 * h.max(),
        #                                   num_peaks=np.inf)
        # lines_polar = list(zip(phi,rho))
        self.lines = pd.DataFrame(np.array(lines).T, columns=['hits', 'angle', 'intercept'])
        self.__divide_line_orientation(self.lines)

    def __divide_line_orientation(self, lines: pd.DataFrame, err=np.pi * 1 / 12, inplace: bool = True):
        if not inplace:
            out = lines.copy()
        else:
            out = lines
        for ix, line in out.iterrows():
            if abs(line['angle']) < err:
                # vertical
                direction = 'v'
            elif abs(line['angle'] - np.pi / 2) < err:
                # horizontal
                direction = 'h'
            else:
                # irrelevant lines
                direction = None
            out._set_value(ix, 'orientation', direction)
        return out

    def intersections(self):
        lines_v = self.lines[self.lines['orientation'] == 'v']
        lines_h = self.lines[self.lines['orientation'] == 'h']
        combinations = list(itertools.product(lines_v.iterrows(), lines_h.iterrows()))

        # intersections = pd.DataFrame()
        for (_, line_h), (_, line_v) in combinations:
            # line_h = lines_h.loc[ix_h]
            # line_v = lines_v.loc[ix_v]
            # TODO mark line
            # point['line_v'] = ix_v
            # point['line_h'] = ix_h
            intersections, point_h, point_v = calc_intersections(line_h, line_v, self.edges_img)
            conn = connectivity(intersections, point_h, point_v, self.edges_img)
            # intersections = intersections.append(point)
        # intersections = intersections.reset_index().drop(['index'], axis=1)
        # return intersections

    def connectivity(self):
        pass

    def corner_detection(self):
        pass

    def warp(self):
        pass
