from typing import Union

import cv2
import numpy as np
import pandas as pd


def intersection_cartesian(L1: pd.DataFrame, L2: pd.DataFrame):
    """
    Compute cartesian coordinates of intersection points given two list of lines in general form.
    General form for a line: Ax+By+C=0

    :param L1:
    :param L2:
    :return:
    """
    if not {"A", "B", "C"}.issubset(set(L1.columns)) or not {"A", "B", "C"}.issubset(set(L2.columns)):
        raise ValueError("L1 and L2 should both contains columns A, B and C, which depicts lines in general form")
    d = L1["A"] * L2["B"] - L1["B"] * L2["A"]
    dx = L1["B"] * L2["C"] - L1["C"] * L2["B"]
    dy = L1["C"] * L2["A"] - L1["A"] * L2["C"]
    x = dx / d
    y = dy / d
    return list(zip(x.values.tolist(), y.values.tolist()))


def points2line(p1, p2):
    """
    Compute Ax+By+C=0 given a list of point [(x1,y1)] and [(x2,y2)].
    Single point is also acceptable.
    :param p1: point in tuple or array (x1,y1) or a list of points in tuple or array [(x1_1,y1_1),(x1_2,y1_2),...]
    :param p2: point in tuple or array (x2,y2) or a list of points in tuple or array [(x2_1,y2_1),(x2_2,y2_2),...]
    :return: pd.DataFrame objects of lines in general form(Ax+By+C=0)
    """
    p1 = np.array(p1)
    p2 = np.array(p2)
    if p1.dtype == np.object_ or p2.dtype == np.object_:
        raise ValueError("p1 and p2 should matrix alike")
    elif len(p1.shape) == 2 and len(p2.shape) == 2:
        if p1.shape[1] != 2 or p2.shape[1] != 2:
            raise ValueError("p1 and p2 should be matrix with column size of exactly 2")
    elif len(p1.shape) == 1 and len(p1) == 2 and len(p1.shape) == 1 and len(p2) == 2:
        p1 = p1.reshape(-1, 2)
        p2 = p2.reshape(-1, 2)
    else:
        raise ValueError("Invalid p1 and p2")

    a = p1[:, 1] - p2[:, 1]
    b = p2[:, 0] - p1[:, 0]
    c = p1[:, 0] * p2[:, 1] - p2[:, 0] * p1[:, 1]
    return pd.DataFrame([a, b, c], index=["A", "B", "C"]).T


def find_y_on_lines(lines: np.array, x: np.array):
    """
    find y of a list of x on a list of lines that in polar form.
    :param lines:
    :param x:
    :return: a list of points, 1th dimension for different x and 2th dimension for different lines
    """
    if len(lines) == 0:
        return lines
    lines = np.array(lines)
    if lines.dtype == np.object_:
        raise ValueError("lines should be matrix alike")
    elif len(lines.shape) == 1:
        if len(lines) == 2:
            lines = lines.reshape(-1, 2)
        else:
            raise ValueError("the length of line vector should 2")
    elif len(lines.shape) == 2:
        if lines.shape[1] != 2:
            raise ValueError("lines should be matrix with column size of exactly 2")
    else:
        raise ValueError("Invalid lines")

    x = np.array(x)
    if x.dtype == np.object_:
        raise ValueError("x should be matrix alike")
    rho = lines[:, 1].reshape(-1, 1)
    phi = lines[:, 0].reshape(-1, 1)
    y = (rho - x * np.cos(phi)) / np.sin(phi)
    return y


def find_points_on_lines(lines: np.array, x: np.array):
    """
    find points of a list of x on a list of lines that in polar form.
    :param lines:
    :param x:
    :return: a list of points, 1th dimension for different x and 2th dimension for different lines
    """
    if len(lines) == 0:
        return lines

    lines = np.array(lines)
    if len(lines.shape) == 1:
        if len(lines) == 2:
            lines = lines.reshape(-1, 2)
    x = np.array(x)

    y = find_y_on_lines(lines, x)
    points = list()
    for ix in range(len(x)):
        points_on_a_line = np.zeros((len(lines), 2))
        points_on_a_line[:, 0] = x[ix]
        points_on_a_line[:, 1] = y[:, ix]
        points.append(list(map(lambda x: tuple(x), points_on_a_line.tolist())))
    return points


def interpolate_pixels_along_line(
    p1: Union[np.ndarray, tuple[int, int]], p2: Union[np.ndarray, tuple[int, int]], width: int = 2
):
    """Uses Xiaolin Wu's line algorithm to interpolate all of the pixels along a
    straight line, given two points (x0, y0) and (x1, y1)

    Wikipedia article containing pseudo code that function was based off of:
        http://en.wikipedia.org/wiki/Xiaolin_Wu's_line_algorithm

    Given by Rick(https://stackoverflow.com/users/2025958/rick)
    on https://stackoverflow.com/questions/24702868/python3-pillow-get-all-pixels-on-a-line.
    """
    if type(p1) is np.ndarray and type(p2) is np.ndarray:
        (x1, y1) = p1.flatten()
        (x2, y2) = p2.flatten()
    elif len(p1) == 2 and len(p2) == 2:
        (x1, y1) = p1
        (x2, y2) = p2
    else:
        raise TypeError("p1 and p2 must be tuple or ndarray depicting points")

    pixels = []
    steep = np.abs(y2 - y1) > np.abs(x2 - x1)

    # Ensure that the path to be interpolated is shallow and from left to right
    if steep:
        t = x1
        x1 = y1
        y1 = t

        t = x2
        x2 = y2
        y2 = t

    if x1 > x2:
        t = x1
        x1 = x2
        x2 = t

        t = y1
        y1 = y2
        y2 = t

    dx = x2 - x1
    dy = y2 - y1
    gradient = dy / dx  # slope

    # Get the first given coordinate and add it to the return list
    x_end = np.round(x1)
    y_end = y1 + (gradient * (x_end - x1))
    xpxl0 = x_end
    ypxl0 = np.round(y_end)
    if steep:
        pixels.extend([(ypxl0, xpxl0), (ypxl0 + 1, xpxl0)])
    else:
        pixels.extend([(xpxl0, ypxl0), (xpxl0, ypxl0 + 1)])

    interpolated_y = y_end + gradient

    # Get the second given coordinate to give the main loop a range
    x_end = np.round(x2)
    y_end = y2 + (gradient * (x_end - x2))
    xpxl1 = x_end
    ypxl1 = np.round(y_end)

    # Loop between the first x coordinate and the second x coordinate, interpolating the y coordinates
    for x in np.arange(xpxl0 + 1, xpxl1):
        if steep:
            pixels.extend([(np.floor(interpolated_y) + i, x) for i in range(1 - width, width + 1)])

        else:
            pixels.extend([(x, np.floor(interpolated_y) + i) for i in range(1 - width, width + 1)])

        interpolated_y += gradient

    # Add the second given coordinate to the given list
    if steep:
        pixels.extend([(ypxl1, xpxl1), (ypxl1 + 1, xpxl1)])
    else:
        pixels.extend([(xpxl1, ypxl1), (xpxl1, ypxl1 + 1)])

    # convert to int
    return list(map(lambda x: tuple(x), np.array(pixels, dtype=np.int_)))


def __order_points(pts):
    pts = np.array(pts)
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect


def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = __order_points(pts)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped
