import cv2
import numpy as np
import pandas as pd
import itertools
from dataclasses import dataclass
from skimage.transform import hough_line_peaks, hough_line
from doc_scanner.math_utils import points2line, find_y_on_lines, intersection, interpolate_pixels_along_line, \
    find_points_on_lines
from matplotlib import pyplot as plt
from doc_scanner.transform import four_point_transform
from doc_scanner.intersection import connectivity
from doc_scanner.transform import four_point_transform
from typing import Tuple


class scanner:
    def __init__(self, image: np.array):
        if len(image.shape) != 2:
            raise ValueError("Image should be a single channel 2D array")
        self.image = image

    def scan(self):
        self.preprocess()
        self.hough_transform()
        self.calc_intersections()
        self.calc_connectivity()
        self.detect_corner()
        # return self.warp()

    def preprocess(self, kernel_size=15, intensity_lower=0, intensity_upper=255, canny_lower=10, canny_upper=70,
                   erode_ks=3):
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
        :param erode_ks:
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

        # TODO blur darker part and dim bright part
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

    def hough_transform(self, err=np.pi * 1 / 12, threshold=0.49, ks=3):
        # -------------------- scikit-image hough line transform --------------------
        theta = np.linspace(-np.pi * 1 / 4, np.pi * 3 / 4, 180)
        h, theta, distance = hough_line(self.edges_img, theta)
        hits, phi, rho = hough_line_peaks(h, theta, distance, min_distance=10, min_angle=50,
                                          threshold=threshold * h.max(),
                                          num_peaks=np.inf)
        lines = list(zip(phi, rho))
        # -------------------- OpenCV hough line transform --------------------
        # lines = cv2.HoughLines(self.edges_img.astype(np.uint8), 1, np.pi / 180, threshold).reshape(-1, 2)
        # lines = list(map(lambda x: tuple(x), lines[:, ::-1].tolist()))

        # -------------------- discriminate between vertical and horizontal lines --------------------
        self.lines = dict(v=[], h=[], o=[])
        for ix, (phi, _) in enumerate(lines):
            if abs(phi) < err or abs(phi - np.pi) < err:
                # vertical
                self.lines['v'].append(lines[ix])
            elif abs(phi - np.pi / 2) < err:
                # horizontal
                self.lines['h'].append(lines[ix])
            else:
                # irrelevant lines
                self.lines['o'].append(lines[ix])
        return self.lines

    def calc_intersections(self):
        """ Compute connectivity given a horizontal line and vertical line in polar coordination.
        1. convert lines to cartesian coordination
        2. find intersection in cartesian coordination
        3.

        :return:
        """
        combinations = list(itertools.product(self.lines['v'], self.lines['h']))
        if len(combinations) == 0:
            self.intersections = pd.DataFrame(columns=['v', 'h', 'cross'])
            return self.intersections

        pairs = np.array(combinations)
        lines_h = pairs[:, 0, :]
        lines_v = pairs[:, 1, :]

        x = (0, self.edges_img.shape[1])

        points_h = find_points_on_lines(lines_h, x)
        points_v = find_points_on_lines(lines_v, x)

        # TODO better bridge between calc_intersections and connectivity
        self.intersections = pd.DataFrame(combinations, columns=['v', 'h'])
        cross = intersection(points2line(*points_h), points2line(*points_v))
        self.intersections = self.intersections.assign(cross=cross)
        return self.intersections

    def calc_connectivity(self):
        x = (0, self.edges_img.shape[1])
        y_h = find_y_on_lines(self.intersections['h'].tolist(), x)
        y_v = find_y_on_lines(self.intersections['v'].tolist(), x)

        if points_h['x'].diff()[1] < 0:
            points_h = points_h.iloc[::-1]
        if points_v['y'].diff()[1] < 0:
            points_v = points_v.iloc[::-1]
        edge_points = pd.DataFrame(columns=['x', 'y'])
        edge_points = edge_points.append(points_h, ignore_index=True)
        edge_points = edge_points.append(points_v, ignore_index=True)
        edge_points.index = pd.Index(['left', 'right', 'top', 'bottom'])

        connectivity = dict()
        for direction, point in edge_points.iterrows():
            distance = np.sqrt(
                (point['y'] - intersections['y']) ** 2 + (point['x'] - intersections['x']) ** 2)[0]
            ratio = along_length / distance
            end = np.round((1 - ratio) * intersections + ratio * point)
            pixels = interpolate_pixels_along_line(intersections, end, width)

            # calculate the numbers of pixels that is not 0 in contour mask
            # and that is within the contour mask image
            hits = 0.0
            pixels_within_image_num = 0.0
            for ix, pixel in pixels.iterrows():
                try:
                    if contour_image[pixel['y'], pixel['x']] > 0:
                        hits += 1
                    pixels_within_image_num += 1
                except IndexError:
                    # TODO when pixels within image are rare, this may introduce false connectivity
                    pass

            connectivity[direction] = dict(hits=hits, pixels_within=pixels_within_image_num,
                                           connectivity=hits / len(pixels))
        connectivity = pd.DataFrame(connectivity).T.sort_values(by='connectivity', ascending=False)
        corner_connectivity = calc_corner_connectivity(connectivity)
        intersections = intersections.assign(**corner_connectivity)

        return intersections

    def detect_corner(self):
        pass

    def warp(self):
        # TODO auto compute corners
        if not hasattr(self, 'corners'):
            raise KeyError('make sure corners has been detected before warp')
        four_point_transform(self.image, self.corners)

    def plot_lines(self, ax):
        x = (0, self.image.shape[1])
        for orientation, lines in self.lines.items():
            y = find_y_on_lines(lines, x)
            if orientation == 'v':
                color = 'r'
            elif orientation == 'h':
                color = 'g'
            else:
                color = 'k'
            for _y in y:
                ax.plot(x, _y, '-{}'.format(color))

    def focus_on_intersection(self, intersection, ax, size=50):
        """Zoom in to have a close look on given intersection

        :param intersection:
        :param ax:
        :param size:
        :return:
        """
        ax.set_xlim((intersection[0] - size, intersection[0] + size))
        ax.set_ylim((intersection[1] - size, intersection[1] + size))

    def reset_plot_view(self, ax):
        ax.set_xlim((0, self.image.shape[1]))
        ax.set_ylim((self.image.shape[0], 0))

    def plot_around_intersection(self, intersection: Tuple[np.int8, np.int8], edges=True, size=50, ax=None):
        """plot image around given intersection

        :param intersection:
        :param edges:
        :param size:
        :return:
        """
        # TODO check intersection inside image
        if ax == None:
            ax = plt.figure().axes

        if edges:
            ax.imshow(self.edges_img)
        else:
            ax.imshow(self.image)

        ax.plot(intersection, 'cx', ms=20)
        ax.set_xlim((intersection[0] - size, intersection[0] + size))
        ax.set_ylim((intersection[1] - size, intersection[1] + size))
        ax.set_axis_off()
        ax.set_title('Detected lines(Intensity)')
