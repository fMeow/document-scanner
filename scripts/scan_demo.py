import os
import cv2
import argparse
from matplotlib import cm
from matplotlib import pyplot as plt
from doc_scanner import scanner

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", dest='show', default='cv')
    parser.add_argument("--from_dir", dest='from_dir', default='./data/images/segment')
    args = parser.parse_args()

    files = os.listdir(args.from_dir)
    for file in files:
        filepath = os.path.join(args.from_dir, file)
        if os.path.isdir(filepath):
            continue
        else:
            if not filepath.endswith('jpg'):
                continue
        image = cv2.imread(os.path.join(args.from_dir, file))

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
        intensity.scan()
        saturation.scan()

        if hasattr(intensity.corners, 'corners'):
            print("-------------------- Intensity --------------------")
            intensity.corners.summary()

        if hasattr(saturation.corners, 'corners'):
            print("-------------------- Saturation --------------------")
            saturation.corners.summary()
        if args.show == 'mpl':
            plt.clf()
            plt.ion()

            ax = plt.subplot(3, 2, 1)
            ax.imshow(intensity.edges_img, cm.gray)
            intensity.plot_lines(ax)
            intensity.reset_plot_view(ax)

            ax = plt.subplot(3, 2, 2)
            ax.imshow(saturation.edges_img, cm.gray)
            saturation.plot_lines(ax)
            saturation.reset_plot_view(ax)

            ax = plt.subplot(3, 2, 3)
            ax.imshow(intensity.edges_img_dilated, cm.gray)
            intensity.plot_lines(ax)
            intensity.plot_corners(ax)
            intensity.reset_plot_view(ax)

            ax = plt.subplot(3, 2, 4)
            ax.imshow(saturation.edges_img_dilated, cm.gray)
            saturation.plot_lines(ax)
            saturation.plot_corners(ax)
            saturation.reset_plot_view(ax)

            ax = plt.subplot(3, 2, 5)
            try:
                intensity_warped = intensity.warp(image)
                ax.imshow(intensity_warped)
            except Exception:
                intensity_warped = None

            ax = plt.subplot(3, 2, 6)
            try:
                saturation_warped = saturation.warp(image)
                ax.imshow(saturation_warped)
            except Exception:
                saturation_warped = None

            plt.pause(0.2)
            plt.waitforbuttonpress()

        elif args.show == 'cv':
            cv2.imshow('original', image)
            # cv2.imshow('Gray Scale', gray)
            # cv2.imshow('opening', hist_equalized)
            # cv2.imshow('Intensity Filtered', filtered)

            # cv2.imshow('saturation edge', saturation_result.edges)
            cv2.imshow('intensity edge', intensity.edges_img)
            # cv2.imshow('hue edge', hue_edges)
            # cv2.imshow('intensity', intensity_blurred)
            cv2.imshow('intensity contour', intensity)
            # cv2.imshow('saturation contour', saturation_with_contours)

            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            raise ValueError("--show must be cv or mpl")
