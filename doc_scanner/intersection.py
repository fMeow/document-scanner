import numpy as np
import pandas as pd
from doc_scanner.math_utils import points2line, find_y_on_lines, intersection, interpolate_pixels_along_line


# def calc_intersections(horizontal_polar, vertical_polar, contour_image):
#     """ Compute connectivity given a horizontal line and vertical line in polar coordination.
#     1. convert lines to cartesian coordination
#     2. find intersection in cartesian coordination
#     3.
#
#     :param horizontal_polar:
#     :param vertical_polar:
#     :param contour_image:
#     :param along_length:
#     :return:
#     """
#
#     x = (0, contour_image.shape[1])
#
#     points_h = find_point_polar(horizontal_polar, x)
#     points_v = find_point_polar(vertical_polar, x)
#
#     intersections = intersection(points2line(*points_h), points2line(*points_v))
#     return intersections, points_h, points_v


def connectivity(intersections, points_h, points_v, contour_image, along_length=50, width=3):
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


def calc_corner_connectivity(connectivity):
    """determine the orientation of corner

    :param connectivity:
    :return:
    """
    # TODO justify and improve connectivity criterion
    corner_connectivity = dict()
    for corner in [['top', 'left'], ['top', 'right'], ['bottom', 'left'], ['bottom', 'right']]:
        label = '{}-{}'.format(*corner)
        conn = connectivity.loc[corner, 'connectivity']
        if conn.sum() == 0:
            corner_connectivity[label] = 0
        else:
            corner_connectivity[label] = 2 * (conn.iloc[0] * conn.iloc[1]) / conn.sum()

    return corner_connectivity
