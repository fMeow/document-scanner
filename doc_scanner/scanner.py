import cv2
import pandas as pd
import itertools
from dataclasses import dataclass
from skimage.transform import hough_line_peaks, hough_line
from doc_scanner.math_utils import points2line, find_point_polar, intersection, interpolate_pixels_along_line


@dataclass
class HoughResult:
    h: pd.np.array
    theta: pd.np.array
    distance: pd.np.array


@dataclass
class ProcessingResult:
    """
    blurred ,edges ,contour_image ,hist_equalized are images of the same shape of input image.
    hist is a 1-D array of histogram.
    contours is opencv result of all contours
    hough is the result of line hough transformation in image-scikit
    """
    blurred: pd.np.array
    edges: pd.np.array
    contour_image: pd.np.array
    hist_equalized: pd.np.array

    hist: pd.np.array

    contours: pd.np.array

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
    kernel = pd.np.ones((kernel_size, kernel_size), dtype=pd.np.int8)
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
    contour_image = cv2.drawContours(pd.np.zeros(image.shape[0:2]), contours, -1, (128, 255, 0), 3)
    theta = pd.np.linspace(-pd.np.pi * 1 / 4, pd.np.pi * 3 / 4, 180)
    hough = hough_line(contour_image, theta)
    return ProcessingResult(blurred=blurred, edges=edges, contour_image=contour_image, hist_equalized=hist_equalized,
                            hist=hist, hough=HoughResult(*hough), contours=contours)


def select_edge(result: ProcessingResult, ax=None, image: pd.np.array = None):
    lines = hough_line_peaks(result.hough.h, result.hough.theta, result.hough.distance, min_distance=10, min_angle=50,
                             threshold=0.5 * result.hough.h.max(), num_peaks=pd.np.inf)
    lines = pd.DataFrame(pd.np.array(lines).T, columns=['hits', 'angle', 'intercept'])
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


def _divide_line_orientation(lines: pd.DataFrame, err=pd.np.pi * 1 / 12, inplace: bool = True):
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
        elif abs(line['angle'] - pd.np.pi / 2) < err:
            # horizontal
            direction = 'horizontal'
        else:
            # irrelevant lines
            direction = 'irrelevant'
        out._set_value(ix, 'direction', direction)
    return out


def _intersection_connectivity(horizontal_polar, vertical_polar, contour_image, along_length=50):
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

    points_h = pd.DataFrame(list(zip(x, y_h)), columns=['x', 'y'], dtype=float)
    points_v = pd.DataFrame(list(zip(x, y_v)), columns=['x', 'y'], dtype=float)

    intersection_point = intersection(points2line(points_h), points2line(points_v))

    if points_h['x'].diff()[1] < 0:
        points_h = points_h.iloc[::-1]
    if points_v['y'].diff()[1] < 0:
        points_v = points_v.iloc[::-1]
    edge_points = pd.DataFrame(columns=['x', 'y'])
    edge_points = edge_points.append(points_h, ignore_index=True)
    edge_points = edge_points.append(points_v, ignore_index=True)
    edge_points.index = pd.Index(['left', 'right', 'down', 'top'])

    connectivity = dict()
    for direction, point in edge_points.iterrows():
        distance = pd.np.sqrt(
            (point['y'] - intersection_point['y']) ** 2 + (point['x'] - intersection_point['x']) ** 2)[0]
        if distance >= along_length:
            ratio = along_length / distance
            end = pd.np.round((1 - ratio) * intersection_point + ratio * point)
            pixels = interpolate_pixels_along_line(intersection_point, end)
            hits = 0
            for ix, pixel in pixels.iterrows():
                if contour_image[pixel['y'], pixel['x']] > 0:
                    hits += 1
            # TODO justify and improve connectivity criterion
            connectivity[direction] = dict(hits=hits, pixel_along=len(pixels), connectivity=hits / len(pixels))
    connectivity = pd.DataFrame(connectivity).T.sort_values(by='connectivity', ascending=False)

    return intersection_point, connectivity


def _corner_orientation(connectivity):
    """determine the orientation of corner

    :param connectivity:
    :return:
    """
    top2_orientations = connectivity.iloc[0:2].index
    if {'top', 'left'} == set(top2_orientations):
        pass
    elif {'top', 'right'} == set(top2_orientations):
        pass
    elif {'down', 'right'} == set(top2_orientations):
        pass
    elif {'down', 'right'} == set(top2_orientations):
        pass


def _find_intersections(lines, contour_image):
    try:
        lines['direction']
    except ValueError:
        _divide_line_orientation(lines)
    vertical_lines = lines[lines['direction'] == 'vertical']
    horizontal_lines = lines[lines['direction'] == 'horizontal']
    combinations = list(itertools.product(vertical_lines.iterrows(), horizontal_lines.iterrows()))
    intersections = pd.DataFrame()
    for (_, vertical_line), (_, horizontal_line) in combinations:
        point, connectivity = _intersection_connectivity(horizontal_line, vertical_line, contour_image)
        intersections = intersections.append(point)
    return intersections
