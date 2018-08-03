import cv2
import numpy as np
import pandas as pd
import itertools
from dataclasses import dataclass
from skimage.transform import hough_line_peaks, hough_line
from doc_scanner.math_utils import points2line, find_point_polar, intersection, interpolate_pixels_along_line
from doc_scanner.transform import four_point_transform


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


def select_edge(result: ProcessingResult, image: np.array = None):
    lines = hough_line_peaks(result.hough.h, result.hough.theta, result.hough.distance, min_distance=10, min_angle=50,
                             threshold=0.49 * result.hough.h.max(), num_peaks=np.inf)
    lines = pd.DataFrame(np.array(lines).T, columns=['hits', 'angle', 'intercept'])
    _divide_line_orientation(lines)
    intersections = _find_intersections(lines, result.contour_image)
    corners = _find_corner(intersections)
    if len(corners) > 0:
        edges = pd.DataFrame(corners).sort_values(by=['score']).iloc[0]
        warped = four_point_transform(image,
                                      np.array([edges['top-left'], edges['top-right'], edges['down-right'], edges['down-left']]))
    else:
        edges = None
        warped = None
    # print(len(corners))

    return lines, intersections, edges, warped


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


def _intersection_connectivity(horizontal_polar, vertical_polar, contour_image, along_length=50, width=3):
    """ Compute connectivity given a horizontal line and vertical line in polar coordination.
    1. convert lines to cartesian coordination
    2. find intersection in cartesian coordination
    3.

    :param horizontal_polar:
    :param vertical_polar:
    :param contour_image:
    :param along_length:
    :return:
    """

    x = (0, contour_image.shape[1])

    y_h = find_point_polar(horizontal_polar, x)
    y_v = find_point_polar(vertical_polar, x)

    points_h = pd.DataFrame(list(zip(x, y_h)), columns=['x', 'y'], dtype=np.float)
    points_v = pd.DataFrame(list(zip(x, y_v)), columns=['x', 'y'], dtype=np.float)

    intersection_point = intersection(points2line(points_h), points2line(points_v))

    if points_h['x'].diff()[1] < 0:
        points_h = points_h.iloc[::-1]
    if points_v['y'].diff()[1] < 0:
        points_v = points_v.iloc[::-1]
    edge_points = pd.DataFrame(columns=['x', 'y'])
    edge_points = edge_points.append(points_h, ignore_index=True)
    edge_points = edge_points.append(points_v, ignore_index=True)
    edge_points.index = pd.Index(['left', 'right', 'top', 'down'])

    connectivity = dict()
    for direction, point in edge_points.iterrows():
        distance = np.sqrt(
            (point['y'] - intersection_point['y']) ** 2 + (point['x'] - intersection_point['x']) ** 2)[0]
        ratio = along_length / distance
        end = np.round((1 - ratio) * intersection_point + ratio * point)
        pixels = interpolate_pixels_along_line(intersection_point, end, width)

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
    corner_connectivity = _corner_orientation(connectivity)
    intersection_point = intersection_point.assign(**corner_connectivity)

    return intersection_point


def _corner_orientation(connectivity):
    """determine the orientation of corner

    :param connectivity:
    :return:
    """
    # TODO justify and improve connectivity criterion
    corner_connectivity = dict()
    for corner in [['top', 'left'], ['top', 'right'], ['down', 'left'], ['down', 'right']]:
        label = '{}-{}'.format(*corner)
        conn = connectivity.loc[corner, 'connectivity']
        if conn.sum() == 0:
            corner_connectivity[label] = 0
        else:
            corner_connectivity[label] = 2 * (conn.iloc[0] * conn.iloc[1]) / conn.sum()

    return corner_connectivity


def _find_intersections(lines, contour_image):
    try:
        lines['direction']
    except ValueError:
        _divide_line_orientation(lines)
    lines_v = lines[lines['direction'] == 'vertical']
    lines_h = lines[lines['direction'] == 'horizontal']
    combinations = list(itertools.product(lines_v.index, lines_h.index))
    intersections = pd.DataFrame()
    for ix_v, ix_h in combinations:
        line_h = lines_h.loc[ix_h]
        line_v = lines_v.loc[ix_v]
        point = _intersection_connectivity(line_h, line_v, contour_image, 100, 2)
        point['line_v'] = ix_v
        point['line_h'] = ix_h
        intersections = intersections.append(point)
    intersections = intersections.reset_index().drop(['index'], axis=1)
    return intersections


def _find_corner(intersections, threshold=0.4):
    """
    label is exactly opposite to corner orientation
    :param intersections:
    :param threshold:
    :return:
    """
    if len(intersections) == 0:
        return list()

    corner = dict()
    count = dict()
    for vertical, horizontal in itertools.product(('top', 'down'), ('left', 'right')):
        label = '{}-{}'.format(_revert_orientation(vertical), _revert_orientation(horizontal))
        # TODO simplify
        points = intersections[(intersections[label] > threshold)]
        corner[label] = points
        count[label] = len(points)
    count = pd.Series(count)

    result = list()
    start = count.idxmin()
    vertical, horizontal = start.split('-')
    for _, point in corner[start].iterrows():
        label_ov = '{}-{}'.format(_revert_orientation(vertical), horizontal)
        ov = corner[label_ov]
        possible_v = ov[ov['line_v'] == point['line_v']]

        label_oh = '{}-{}'.format(vertical, _revert_orientation(horizontal))
        oh = corner[label_oh]
        possible_h = oh[oh['line_h'] == point['line_h']]

        for (_, v), (_, h) in itertools.product(possible_v.iterrows(), possible_h.iterrows()):

            label_o = '{}-{}'.format(_revert_orientation(vertical), _revert_orientation(horizontal))
            o = corner[label_o]
            for _, point_o in o.iterrows():
                if point_o['line_h'] == v['line_h'] and point_o['line_v'] == h['line_v']:
                    score = (point[start] + v[label_ov] + h[label_oh] + point_o[label_o]) / 4
                    result.append(
                        {start: point.loc[['x', 'y']].values.tolist(), label_ov: v.loc[['x', 'y']].values.tolist(),
                         label_oh: h.loc[['x', 'y']].values.tolist(), label_o: point_o.loc[['x', 'y']].values.tolist(),
                         'score': score})

    return result


def _revert_orientation(orientation):
    if orientation == 'left':
        return 'right'
    elif orientation == 'right':
        return 'left'
    elif orientation == 'top':
        return 'down'
    elif orientation == 'down':
        return 'top'
