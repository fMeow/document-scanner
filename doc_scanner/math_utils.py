import numpy as np
import pandas as pd


def intersection(L1: pd.DataFrame, L2: pd.DataFrame):
    """
    Compute intersection given two lines in general form.
    General form for a line: Ax+By+C=0

    :param L1:
    :param L2:
    :return:
    """
    if not {'A', 'B', 'C'}.issubset(set(L1.columns)) or not {'A', 'B', 'C'}.issubset(set(L2.columns)):
        raise ValueError('L1 and L2 should both contains columns A, B and C, which depicts lines in general form')
    d = (L1['A'] * L2['B'] - L1['B'] * L2['A']).iloc[0]
    dx = (L1['C'] * L2['B'] - L1['B'] * L2['C']).iloc[0]
    dy = (L1['A'] * L2['C'] - L1['C'] * L2['A']).iloc[0]
    x = dx / d
    y = dy / d
    return pd.DataFrame([[x, y]], columns=['x', 'y'])


def points2line(points: pd.DataFrame):
    """
    compute Ax+By+C=0 given point (x1,y1) and (x2,y2)
    :param p1:
    :param p2:
    :return:
    """
    if len(points) != 2:
        raise ValueError('points should only exactly 2 points')
    elif 'x' not in points.columns or 'y' not in points.columns:
        raise ValueError('points should contain columns x and y')

    a = points['y'].diff()[1]
    b = -points['x'].diff()[1]
    c = points.loc[0, 'x'] * points.loc[1, 'y'] - points.loc[1, 'x'] * points.loc[0, 'y']
    # a = (p1[1] - p2[1])
    # b = (p2[0] - p1[0])
    # c = (p1[0] * p2[1] - p2[0] * p1[1])
    return pd.DataFrame([[a, b, c]], columns=['A', 'B', 'C'])


def find_point_polar(line: pd.DataFrame, x: tuple):
    angle = line['angle']
    dist = line['intercept']
    y = tuple(map(lambda i: (dist - i * np.cos(angle)) / np.sin(angle), x))
    return y


def interpolate_pixels_along_line(p1: pd.DataFrame, p2: pd.DataFrame):
    """Uses Xiaolin Wu's line algorithm to interpolate all of the pixels along a
    straight line, given two points (x0, y0) and (x1, y1)

    Wikipedia article containing pseudo code that function was based off of:
        http://en.wikipedia.org/wiki/Xiaolin_Wu's_line_algorithm

    Given by Rick(https://stackoverflow.com/users/2025958/rick)
    on https://stackoverflow.com/questions/24702868/python3-pillow-get-all-pixels-on-a-line.
    """
    (x1, y1) = p1.values.flatten()
    (x2, y2) = p2.values.flatten()
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
            pixels.extend([(np.floor(interpolated_y), x), (np.floor(interpolated_y) + 1, x)])

        else:
            pixels.extend([(x, np.floor(interpolated_y)), (x, np.floor(interpolated_y) + 1)])

        interpolated_y += gradient

    # Add the second given coordinate to the given list
    if steep:
        pixels.extend([(ypxl1, xpxl1), (ypxl1 + 1, xpxl1)])
    else:
        pixels.extend([(xpxl1, ypxl1), (xpxl1, ypxl1 + 1)])

    return pd.DataFrame(pixels, columns=['x', 'y'], dtype=int)
