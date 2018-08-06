import numpy as np
from functools import total_ordering
from doc_scanner.math_utils import interpolate_pixels_along_line, find_points_on_lines


class Intersection:
    ORDER = ['top', 'right', 'bottom', 'left']
    ORIENTATION_ORDER = ['top-left', 'top-right', 'bottom-right', 'bottom-left']

    def __init__(self, intersection, line_v, line_h, x=None, image=None, along_length=50, width=3):
        """

        :param line_v: polar form
        :param line_h: polar form
        """
        if len(intersection) != 2:
            raise ValueError("intersection must be a tuple likes (x,y)")
        elif len(line_v) != 2:
            raise ValueError("line_v must be a tuple likes (phi,rho)")
        elif len(line_h) != 2:
            raise ValueError("line_h must be a tuple likes (phi,rho)")

        self.intersection = intersection
        self.line_v = line_v
        self.line_h = line_h
        self.__connectivity = None
        self.__orientation = None
        self.along_length = along_length
        self.width = width
        # manually assgin x source and destination or decide x automatically if given image
        if image is None and x is None:
            raise ValueError("Either a tuple of x coordinates or image must be set")
        elif image:
            self.image = image
            self.x = image.shape[1]
        else:
            self.image = None
            self.x = x

        self.corners = self.__corner_points()

    def __corner_points(self):
        x = self.x
        p_h = find_points_on_lines(self.line_h, x)
        # get a list of points on horizontal edges
        p_h = list(map(lambda x: tuple(x), np.array(p_h).reshape(-1, 2)))

        p_v = find_points_on_lines(self.line_v, x)
        p_v = list(map(lambda x: tuple(x), np.array(p_v).reshape(-1, 2)))

        # points[point][0:x,1:y]
        if p_h[0][0] < p_h[1][0]:
            left = p_h[0]
            right = p_h[1]
        else:
            left = p_h[1]
            right = p_h[0]

        if p_v[0][1] > p_v[1][1]:
            top = p_v[0]
            bottom = p_v[1]
        else:
            top = p_v[1]
            bottom = p_v[0]

        return top, right, bottom, left

    def connectivity(self, along_length=None, width=None):
        validate_along_length = along_length is None or along_length == self.along_length
        validate_width = width is None or width == self.width
        if self.__connectivity is not None and validate_along_length and validate_width:
            return self.__connectivity
        else:
            if along_length:
                self.along_length = along_length
            if width:
                self.width = width
            hits = [0] * 4
            longth = [0] * 4
            ends = [0] * 4
            for ix, point in enumerate(self.corners):
                point = np.array(point).reshape(1, 2)
                intersection = np.array(self.intersection).reshape(1, 2)
                # l2 norm (euclidean distance) of difference
                distance = np.linalg.norm(point - intersection)
                ratio = self.along_length / distance
                ends[ix] = np.round((1 - ratio) * intersection + ratio * point).astype(np.int)
                pixels = interpolate_pixels_along_line(intersection, ends[ix], self.width)

                # calculate the numbers of pixels that is not 0 in contour mask
                # and that is within the contour mask image
                for pixel in pixels:
                    try:
                        if self.image[pixel[1], pixel[0]] > 0:
                            hits[ix] += 1
                        longth[ix] += 1
                    except IndexError:
                        # TODO when pixels within image are rare, this may introduce false connectivity
                        pass
            self.hits = hits
            self.longth = longth
            self.ends = ends
            self.__connectivity = tuple(np.array(hits) / np.array(self.longth))
            return self.__connectivity

    def orientation(self, ):
        if self.__orientation:
            return self.__orientation
        else:
            connectivity = self.connectivity()
            scores = []
            for v, h in ((0, 1), (0, 3), (2, 3), (2, 1)):
                c1 = connectivity[v]
                c2 = connectivity[h]
                summation = c1 + c2
                scores.append(2 * (c1 * c2) / summation if summation != 0 else 0)

            self.__orientation = scores
            return self.__orientation

    def __repr__(self):
        return 'Cross at: ({:.2f}, {:.2f})'.format(*self.intersection)

    def summary(self, one_line=False, println=True):
        summary = 'Cross at: ({:.2f}, {:.2f})'.format(*self.intersection)

        if not one_line:
            summary += '\n'
        else:
            summary += '\t'

        summary += 'Connectivity: '
        connectivity = self.connectivity()
        for i in range(4):
            summary += '{:.4f}({}/{}), '.format(connectivity[i], self.hits[i], self.longth[i])

        if not one_line:
            summary += '\n'
        else:
            summary += '\t'

        summary += 'Orientation scores:'
        scores = self.orientation()
        for i in range(4):
            summary += '{:.4f}, '.format(scores[i])

        if not one_line:
            summary += '\n'
        else:
            summary += '\t'

        if println:
            print(summary)
        else:
            return summary


@total_ordering
class Frame:
    ORIENTATION_ORDER = ['top-left', 'top-right', 'bottom-right', 'bottom-left']

    def __init__(self, top_left: Intersection, top_right: Intersection, bottom_right: Intersection,
                 bottom_left: Intersection, image_shape=None):
        self.corners = (top_left, top_right, bottom_right, bottom_left)
        self.__score = None
        self.image_shape = image_shape

    def __eq__(self, other):
        return self.score() == other.score()

    def __ne__(self, other):
        return self.score() != other.score()

    def __lt__(self, other):
        return self.score() < other.score()

    def score(self, image_shape=None):
        if image_shape == self.image_shape and image_shape is not None:
            raise ValueError("Image area not set yet")
        elif self.__score is not None and image_shape == self.image_shape:
            return self.__score
        else:
            if image_shape:
                self.image_shape = image_shape
            total_area = self.image_shape[0] * self.image_shape[1]
            scores = []
            for i in range(4):
                scores.append(self.corners[i].orientation()[i] * self.area() / total_area)
            self.__score = sum(scores)
            return self.__score

    def area(self):
        (top_left, top_right, bottom_right, bottom_left) = self.corners
        a = np.abs(top_left.intersection[1] - bottom_left.intersection[1])
        b = np.abs(top_left.intersection[0] - top_right.intersection[0])
        return a * b * np.sin(np.abs(top_left.line_h[0] - top_left.line_v[0]))

    def coordinates(self):
        """
        Get coordinates of 4 corners
        :return:
        """
        corners = list()
        for corner in self.corners:
            corners.append(corner.intersection)
        return tuple(corners)

    def __repr__(self):
        return "rectangle"

    def summary(self):
        summary = ""
        for i in range(4):
            summary += "{}: ".format(self.ORIENTATION_ORDER[i])
            summary += self.corners[i].summary(one_line=True, println=False)
            summary += '\n'
        summary += 'Area: {}'.format(self.area())

        print(summary)
