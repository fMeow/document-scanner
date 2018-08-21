# Document-Scanner
[![Build Status](https://travis-ci.org/Guoli-Lyu/document-scanner.svg?branch=master)](https://travis-ci.org/Guoli-Lyu/document-scanner)

Document-Scanner is open-source python package to scan, segment and tranform images of documents as if the documents is scanned by a scanner. It includes predefined pipelines on preprocessing, frame detection, transformation and post processing to add styles.

## Pipeline

1. Convert to HSV color space

    The following pipelines is applied first on **intensity** slice , or the Value phase, of the original image. If failed to find frame in the intensity image, apply exactly the same processes to **saturation** image. 
    
1. Preprocessing
    1. Blur with Median filter
    1. Histogram equalization
    1. Morphological operation (Opening)
    1. (Optional) Threshold based segmentation.

        Here we assume that the document of interest is mainly white while background is darker.
        Then we can extract document from background with a proper threshold.
        After histogram, maybe we can just assume the document lays in the half brighter part on histogram.
    1. Canny edge detector
    1. Contour detection
    1. Morphological Erosion
    1. Morphological Dilation
        
        This step is to dilate the contour to reduce the impact of non-linear edge when calculating connectivity.
        
1. Hough Transform
1. Intersection
    1. Find the cartesian coordination of intersection points
    1. Calculate connectivity on every intersections on four direction: up, right, bottom, left.
1, Corner
    Compute the possiblity on every intersection points to decide the orientation of corner.
1. Frame detection
    1. Find possible frames
    1. Select the most possible frame
1. Warp
1. (**TODO**) Post process

## Demo

Use /scripts/scan_demo.py to see what's happen.

Put images under /data/images and run the scripts.

## Usage

## Dependencies
The minimum required dependencies to run document-scanner are:

-   Python>=3.6
-   openCV3
-   scikit-image
-   pandas
-   numpy

Use the following command to install dependencies with pip:
```bash
$ pip install -r requirements.txt
```

## Contribution

