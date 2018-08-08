from doc_scanner import math_utils
import pandas as pd
import numpy as np
import pytest


def test_points2lines():
    line_ground_truth = pd.DataFrame([[-3, 1, 0], [20, -7, 0], [32, -4, -144]], columns=['A', 'B', 'C'])
    # list of points
    p1 = [(0, 0), (7, 20), (7, 20)]
    p2 = [(1, 3), (0, 0), (3, -12)]
    line1 = math_utils.points2line(p1, p2)
    assert (line1 == line_ground_truth).all().all()

    p1 = np.array([(0, 0), (7, 20), (7, 20)])
    p2 = np.array([(1, 3), (0, 0), (3, -12)])
    line1 = math_utils.points2line(p1, p2)
    assert (line1 == line_ground_truth).all().all()


def test_points2line():
    line_ground_truth = pd.DataFrame([[-3, 1, 0]], columns=['A', 'B', 'C'])
    # single points
    p1 = (0, 0)
    p2 = (1, 3)
    line1 = math_utils.points2line(p1, p2)
    assert np.allclose(line1, line_ground_truth)

    p1 = np.array((0, 0))
    p2 = np.array((1, 3))
    line2 = math_utils.points2line(p1, p2)
    assert np.allclose(line2, line_ground_truth)


def test_points2line_invalid_input():
    p1 = (0, 0, 0)
    p2 = (1, 3)
    with pytest.raises(ValueError):
        math_utils.points2line(p1, p2)

    p1 = (0, 0,)
    p2 = (1, 3, 0)
    with pytest.raises(ValueError):
        math_utils.points2line(p1, p2)

    p1 = [(0, 0, 0), (0, 0)]
    p2 = [(1, 3, 0), (0, 0)]
    with pytest.raises(ValueError):
        math_utils.points2line(p1, p2)

    # 3 dimension
    p1 = np.array([[(0, 0), (0, 0)]])
    p2 = np.array([[(0, 0), (0, 0)]])
    with pytest.raises(ValueError):
        math_utils.points2line(p1, p2)


def test_find_y_on_lines():
    x = (0, 100)
    lines = [(np.pi / 2, 10), (np.pi / 4, np.sqrt(2)), ]
    result = np.array([[10, 10], [2.0, -98.0]])

    y = math_utils.find_y_on_lines(lines, x)
    assert np.allclose(y, result)


def test_find_y_on_line():
    x = (0, 10)
    line = (np.pi / 2, 10)
    result = np.array([[10, 10]])

    y = math_utils.find_y_on_lines(line, x)
    assert np.allclose(y, result)


def test_find_y_on_lines_invalid():
    line = (np.pi / 2, 10, 10)
    with pytest.raises(ValueError):
        math_utils.find_y_on_lines(line, (0, 10))

    line = ((np.pi / 2, 10, 10), (1, 2))
    with pytest.raises(ValueError):
        math_utils.find_y_on_lines(line, (0, 10))

    # the rows of lines should be the same as x
    line = ((np.pi / 2, 10), (1, 2))
    with pytest.raises(ValueError):
        math_utils.find_y_on_lines(line, ((0, 10,), (1, 0), (0, 0)))

    # invalid x
    line = ((np.pi / 2, 10), (1, 2))
    with pytest.raises(ValueError):
        math_utils.find_y_on_lines(line, ((0, 10, 0), (1, 0)))


def test_find_points_on_lines():
    x = (0, 100)
    lines = [(np.pi / 2, 10), (np.pi / 4, np.sqrt(2)), ]
    result = np.array([
        [(0, 10), (0, 2)],
        [(100, 10), (100, -98)],
    ])

    y = math_utils.find_points_on_lines(lines, x)
    assert np.allclose(y, result)


def test_find_points_on_line():
    x = (0, 10)
    line = (np.pi / 2, 10)
    result = np.array([[(0, 10)], [(10, 10)]])

    y = math_utils.find_points_on_lines(line, x)
    assert np.allclose(y, result)


def test_interpolate_pixels_along_line():
    result = [(0, 0), (0, 1), (1, -1), (1, 0), (1, 1), (1, 2), (2, -1), (2, 0), (2, 1), (2, 2), (3, 0), (3, 1), (3, 2),
              (3, 3), (4, 0), (4, 1), (4, 2), (4, 3), (5, 1), (5, 2), (5, 3), (5, 4), (6, 1), (6, 2), (6, 3), (6, 4),
              (7, 3), (7, 4)]
    pixels = math_utils.interpolate_pixels_along_line((0, 0), (7, 3), width=2)
    assert pixels == result
