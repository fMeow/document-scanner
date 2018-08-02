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
