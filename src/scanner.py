import cv2
import os
import numpy as np
from matplotlib import pyplot as plt


def filter_and_edge_detect(image,kernel_size=15,intensity_lower=0,intensity_upper=255,canny_lower=0,canny_upper=100):
    # image = cv2.GaussianBlur(image, (5, 5), 0)
    # image = cv2.medianBlur(image, 15)
    image = cv2.bilateralFilter(image, 9,30,30)
    hist_equalized = cv2.equalizeHist(image)

    # Morphological Open operation
    # Determine kernel size according to a priori knowledge on the size of words
    kernel = np.ones((kernel_size, kernel_size), dtype=np.int8)
    hist_equalized = cv2.morphologyEx(hist_equalized, cv2.MORPH_OPEN, kernel)

    # hist = cv2.calcHist([hist_equalized], [0], None, [256], [0, 256])
    # plt.bar(np.arange(len(hist)), hist.flatten())
    # plt.show()

    # TODO intensity threshold filter can bring artifacts
    # Threshold the intensity image or gray scale image
    # ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    mask = cv2.inRange(hist_equalized, intensity_lower, intensity_upper)

    # Bitwise-AND mask and original image
    filtered = cv2.bitwise_and(hist_equalized, hist_equalized, mask=mask)

    # TODO decide thresholds
    edges = cv2.Canny(filtered, canny_lower, canny_upper, L2gradient=True)
    return edges


if __name__ == '__main__':
    """
    Basic idea:
    1. Convert color space from RGB to HSV
    1. Filter and segmentation
        1. Compute gray scale or so called intensity, which is exactly the V dimension of HSV color space
        1. Histogram equalization
        1. Threshold based segmentation.
        
            Here we assume that the document of interest is mainly white while background is darker.
            Then we can extract document from background with a proper threshold.
            After histogram, maybe we can just assume the document lays in the half brighter part on histogram.
    """
    files = os.listdir('../images')
    for file in files:
        if os.path.isdir(os.path.join('../images', file)):
            continue
        image = cv2.imread(os.path.join('../images', file))
        height, width, _ = image.shape
        if height > width:
            resize_ratio = width / 500
        else:
            resize_ratio = height / 500
        image = cv2.resize(image, (0, 0), fx=1 / resize_ratio, fy=1 / resize_ratio, interpolation=cv2.INTER_AREA)

        # Convert RGB to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        hue = hsv[:, :, 0]
        saturation = hsv[:, :, 1]
        intensity = hsv[:, :, 2]

        hue_edges = filter_and_edge_detect(hue,canny_lower=30,canny_upper=150)
        saturation_edges = filter_and_edge_detect(saturation)
        intensity_edges = filter_and_edge_detect(intensity)

        fused = (hue_edges + intensity_edges + saturation_edges) / 3
        fused = fused.astype(np.uint8)

        # cv2.HoughLines(edges,)

        # im2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # image_with_contours = cv2.drawContours(edges.copy(), contours, -1, (128, 255, 0), 3)

        # plt.clf()
        # plt.ion()
        # ax = plt.subplot(2, 2, 1)
        # ax.imshow(image)
        # ax.set_title("Original")
        #
        # ax = plt.subplot(2, 2, 2)
        # ax.imshow(hue_edges, cmap='gray')
        # ax.set_title("Hue")
        #
        # ax = plt.subplot(2, 2, 3)
        # ax.imshow(intensity_edges, cmap='gray')
        # ax.set_title("Intensity")
        #
        # ax = plt.subplot(2, 2, 4)
        # ax.imshow(saturation_edges, cmap='gray')
        # ax.set_title("Saturation")
        #
        # # plt.tight_layout()
        #
        # plt.pause(0.2)
        # plt.waitforbuttonpress()


        # cv2.imshow('Gray Scale', gray)
        # cv2.imshow('opening', hist_equalized)
        # cv2.imshow('Intensity Filtered', filtered)

        cv2.imshow('saturation',saturation_edges)
        cv2.imshow('intensity',intensity_edges)
        cv2.imshow('hue',hue_edges)
        cv2.imshow('original', image)
        # cv2.imshow('contours', contours)
        # cv2.imshow('contours', image_with_contours)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
