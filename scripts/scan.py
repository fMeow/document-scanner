import os
import cv2
import argparse
import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt
from doc_scanner import scanner
from doc_scanner.math_utils import find_y_on_lines

parser = argparse.ArgumentParser()
parser.add_argument("--show", dest='show', default='mpl')
parser.add_argument("--image-path", dest='image_path', default='./data/images')
options = parser.parse_args()

files = os.listdir(options.image_path)
for file in files:
    filepath = os.path.join(options.image_path, file)
    if os.path.isdir(filepath):
        continue
    else:
        if not filepath.endswith('jpg'):
            continue
    image = cv2.imread(os.path.join(options.image_path, file))

    # resize image
    height, width, _ = image.shape
    if height > width:
        resize_ratio = width / 500
    else:
        resize_ratio = height / 500
    image = cv2.resize(image, (0, 0), fx=1 / resize_ratio, fy=1 / resize_ratio, interpolation=cv2.INTER_AREA)

    # Convert RGB to HSV colorspace
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # hue ranges from 0-180
    hue = scanner(hsv[:, :, 0])
    saturation = scanner(hsv[:, :, 1])
    intensity = scanner(hsv[:, :, 2])

    intensity_warped = intensity.scan()
    saturation_warped = saturation.scan()

    plt.clf()
    plt.ion()
    if hasattr(intensity.corners, 'corners'):
        print("-------------------- Intensity --------------------")
        intensity.corners.summary()

    if hasattr(saturation.corners, 'corners'):
        print("-------------------- Saturation --------------------")
        saturation.corners.summary()

    ax = plt.subplot(2, 2, 1)
    ax.imshow(intensity.edges_img, cm.gray)
    intensity.plot_lines(ax)
    intensity.reset_plot_view(ax)

    ax = plt.subplot(2, 2, 2)
    ax.imshow(saturation.edges_img, cm.gray)
    saturation.plot_lines(ax)
    saturation.reset_plot_view(ax)

    ax = plt.subplot(2, 2, 3)
    ax.imshow(intensity.edges_img_dilated, cm.gray)
    intensity.plot_lines(ax)
    intensity.plot_corners(ax)
    intensity.reset_plot_view(ax)

    ax = plt.subplot(2, 2, 4)
    ax.imshow(saturation.edges_img_dilated, cm.gray)
    saturation.plot_lines(ax)
    saturation.plot_corners(ax)
    saturation.reset_plot_view(ax)

    plt.pause(0.2)
    plt.waitforbuttonpress()
    # if options.show == 'mpl':
    #     plt.clf()
    #     plt.ion()
    #     ax = plt.subplot(2, 2, 1)
    #     ax.imshow(image)
    #     ax.set_title("Original")
    #
    #     ax = plt.subplot(2, 2, 2)
    #     ax.imshow(intensity.edges_img, cmap='gray')
    #     ax.set_title("Edges(Intensity)")
    #
    #     ax = plt.subplot(2, 2, 3)
    #     ax.imshow(intensity.edges_img, cmap='gray')
    #     ax.set_title("Edges(Intensity)")
    #
    #     ax = plt.subplot(2, 2, 4)
    #     if warped is not None:
    #         ax.imshow(warped)
    #
    #     ax = plt.subplot(3, 3, 6)
    #     # ax.imshow(np.log(1 + saturation_result.hough.h), cmap=cm.gray, aspect=1 / 1.5,
    #     #           extent=[np.rad2deg(saturation_result.hough.theta[-1]), np.rad2deg(saturation_result.hough.theta[0]),
    #     #                   500, 300],
    #     #           )
    #     ax.set_title('Hough transform(Saturation)')
    #     ax.set_ylabel('Distance (pixels)')
    #     ax.axis('image')
    #
    #     ax = plt.subplot(3, 3, 8)
    #     ax.imshow(image)
    #     if ax and image is not None:
    #         for ix, line in intensity.lines.iterrows():
    #             x = (0, image.shape[1])
    #             y = find_point_polar(line, x)
    #             if line['direction'] == 'vertical':
    #                 color = 'r'
    #             elif line['direction'] == 'horizontal':
    #                 color = 'g'
    #             else:
    #                 color = 'k'
    #             ax.plot(x, y, '-{}'.format(color))
    #
    #         try:
    #             x = intensity.intersections['x'].values
    #             y = intensity.intersections['y'].values
    #             ax.plot(x, y, 'bx', ms=20)
    #         except KeyError:
    #             pass
    #
    #         # if len(corners) > 0:
    #         #     points = np.array(edges.drop('score').values.tolist())
    #         #     ax.plot(points[:, 0], points[:, 1], 'cx', ms=20)
    #     ax.set_xlim((0, image.shape[1]))
    #     ax.set_ylim((image.shape[0], 0))
    #     ax.set_axis_off()
    #     ax.set_title('Detected lines(Intensity)')
    #
    #     # ax = plt.subplot(3, 3, 9)
    #     # ax.imshow(image)
    #     # select_edge(saturation_result, image)
    #     # ax.set_xlim((0, image.shape[1]))
    #     # ax.set_ylim((image.shape[0], 0))
    #     # ax.set_axis_off()
    #     # ax.set_title('Detected lines(Saturation)')
    #
    #     # plt.tight_layout()
    #     plt.pause(0.2)
    #     plt.waitforbuttonpress()
    #
    # elif options.show == 'cv':
    #     # cv2.imshow('Gray Scale', gray)
    #     # cv2.imshow('opening', hist_equalized)
    #     # cv2.imshow('Intensity Filtered', filtered)
    #
    #     # cv2.imshow('saturation edge', saturation_result.edges)
    #     # cv2.imshow('intensity edge', intensity_result.edges)
    #     # cv2.imshow('hue edge', hue_edges)
    #     cv2.imshow('original', image)
    #     # cv2.imshow('intensity', intensity_blurred)
    #     # cv2.imshow('intensity contour', intensity_with_contours)
    #     # cv2.imshow('saturation contour', saturation_with_contours)
    #
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    # else:
    #     raise ValueError("--show must be cv or mpl")
