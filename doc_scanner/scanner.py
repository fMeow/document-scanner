import cv2
import numpy as np
import itertools
from skimage.transform import hough_line_peaks, hough_line
from doc_scanner.math_utils import points2line, find_y_on_lines, intersection_cartesian, find_points_on_lines
from matplotlib import pyplot as plt
from doc_scanner.model import Intersection, Frame
from doc_scanner.transform import four_point_transform


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
                   erode_ks=3, dilate_ks=15):
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
        # -------------------- Dilation --------------------
        kernel = np.ones((dilate_ks, dilate_ks), dtype=np.int8)
        self.edges_img_dilated = cv2.morphologyEx(self.edges_img, cv2.MORPH_DILATE, kernel)
        # -------------------- Erosion --------------------
        # TODO Kernel shape
        kernel = np.ones((erode_ks, erode_ks), dtype=np.int8)
        self.edges_img = cv2.morphologyEx(self.edges_img, cv2.MORPH_ERODE, kernel)

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
            self.intersections = list()
            return self.intersections

        pairs = np.array(combinations)
        lines_h = pairs[:, 0, :]
        lines_v = pairs[:, 1, :]

        x = (0, self.edges_img.shape[1])

        points_h = find_points_on_lines(lines_h, x)
        points_v = find_points_on_lines(lines_v, x)

        cross = intersection_cartesian(points2line(*points_h), points2line(*points_v))

        self.intersections = list()
        for ix in range(len(combinations)):
            self.intersections.append(Intersection(cross[ix], *combinations[ix], x=x))
        return self.intersections

    def calc_connectivity(self):

        for intersection in self.intersections:
            intersection.image = self.edges_img_dilated
            intersection.connectivity()
        return self.intersections

    def detect_corner(self, orientation_score_threshold=0.4, relative_area_threshold=0.3):
        corner = dict()
        for orientation in Intersection.ORIENTATION_ORDER:
            corner[orientation] = list()
        for intersection in self.intersections:
            orientation_score = intersection.orientation()
            for ix in range(4):
                if orientation_score[ix] > orientation_score_threshold:
                    corner[Intersection.ORIENTATION_ORDER[ix]].append(intersection)

        possible_rectangle = list()
        for top_left in corner['top-left']:

            bottom_left_candidates = list()
            for bottom_left in corner['bottom-left']:
                if bottom_left == top_left:
                    continue
                if bottom_left.line_v == top_left.line_v:
                    bottom_left_candidates.append(bottom_left)

            top_right_candidates = list()
            for top_right in corner['top-right']:
                if top_right == top_left:
                    continue
                if top_right.line_h == top_left.line_h:
                    top_right_candidates.append(top_right)

            combinations = list(itertools.product(top_right_candidates, bottom_left_candidates))
            for bottom_right in corner['bottom-right']:
                if bottom_right == top_left:
                    continue
                for top_right, bottom_left in combinations:
                    if bottom_right.line_v == top_right.line_v and bottom_right.line_h == bottom_left.line_h:
                        frame = Frame(top_left, top_right, bottom_right, bottom_left,
                                      image_shape=self.edges_img_dilated.shape)
                        if frame.relative_area() > relative_area_threshold:
                            possible_rectangle.append(frame)

        if len(possible_rectangle) > 0:
            self.corners = max(possible_rectangle)
        else:
            self.corners = None
        return possible_rectangle

    def warp(self, image=None, scale=1):
        if not hasattr(self, 'corners'):
            raise KeyError('make sure corners has been detected before warp')
        if self.corners:
            if image is None:
                image = self.image
            elif not (np.array(image.shape[0:2]) * scale - np.array(self.image.shape[0:2])).any():
                raise ValueError("image should be in shape {}".format(self.image.shape[0:2]))
            self.warped = four_point_transform(image, np.array(self.corners.coordinates()) * scale)
        else:
            raise ValueError("Fail to find possible rectangle")
        return self.warped

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

    def focus_on_intersection(self, intersection: Intersection, ax, size=50):
        """Zoom in to have a close look on given intersection

        :param intersection:
        :param ax:
        :param size:
        :return:
        """
        ax.set_xlim((intersection.intersection[0] - size, intersection.intersection[0] + size))
        ax.set_ylim((intersection.intersection[1] - size, intersection.intersection[1] + size))

    def reset_plot_view(self, ax):
        ax.set_xlim((0, self.image.shape[1]))
        ax.set_ylim((self.image.shape[0], 0))

    def plot_corners(self, ax):
        if self.corners:
            for corner in self.corners.coordinates():
                ax.plot(corner[0], corner[1], 'cx', ms=20)

    def plot_around_intersection(self, intersection: Intersection, edges=True, size=50, ax=None):
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

        ax.plot(intersection.intersection, 'cx', ms=20)
        ax.set_xlim((intersection.intersection[0] - size, intersection.intersection[0] + size))
        ax.set_ylim((intersection.intersection[1] - size, intersection.intersection[1] + size))
        ax.set_axis_off()
        ax.set_title('Detected lines(Intensity)')
