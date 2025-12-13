# Document-Scanner

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
1. Intersection 1. Find the cartesian coordination of intersection points 1. Calculate connectivity on every intersections on four direction: up, right, bottom, left.
   1, Corner
   Compute the possiblity on every intersection points to decide the orientation of corner.
1. Frame detection
   1. Find possible frames
   1. Select the most possible frame
1. Warp
1. (**TODO**) Post process


## Installation

Install directly from GitHub:

```bash
pip install git+https://github.com/dantetemplar/updated-fMeow-document-scanner
```

Or using uv:

```bash
uv pip install git+https://github.com/dantetemplar/updated-fMeow-document-scanner
```

The minimum required dependencies to run document-scanner are:

- Python>=3.9
- OpenCV4
- scikit-image
- pandas
- numpy>=2.0

## Usage

### Basic Example

```python
import cv2
from doc_scanner import scanner

# Load an image
image = cv2.imread("document.jpg")

# Convert to HSV color space
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Create scanner instance with intensity channel (Value)
intensity_scanner = scanner(hsv[:, :, 2])

# Run the scanning pipeline
intensity_scanner.scan()

# Check if corners were detected
if intensity_scanner.corners is not None:
    # Warp the image to extract the document
    warped = intensity_scanner.warp(image)
    cv2.imwrite("scanned_document.jpg", warped)
else:
    # Try with saturation channel as fallback
    saturation_scanner = scanner(hsv[:, :, 1])
    saturation_scanner.scan()
    if saturation_scanner.corners is not None:
        warped = saturation_scanner.warp(image)
        cv2.imwrite("scanned_document.jpg", warped)
```


### Command-Line Script

For batch processing, use the provided script (you can copy it to your project and run it):

```bash
uv run scripts/scan.py --from_dir ./data/images/segment --to_dir ./output
```

## Contributing

Contributions are welcome! This project uses [uv](https://github.com/astral-sh/uv) for dependency management, [ruff](https://github.com/astral-sh/ruff) for linting and formatting. Configuration is in `pyproject.toml`.

### Development Setup

1. **Fork the repository and then clone it to your local machine**.
1. **Install uv** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
1. **Create a feature branch (`git checkout -b feature/amazing-feature`)**
1. **Install the project in editable mode with dev dependencies**:
   ```bash
   uv pip install -e ".[dev]"
   ```
1. **Make your changes and ensure code is formatted and tests pass**:

      **Run linting**:
      ```bash
      uv run ruff check . --fix
      ```

      **Format code**:
      ```bash
      uv run ruff format .
      ```

      **Run tests**:
      ```bash
      uv run pytest
      ```

1. **Commit your changes (`git commit -m 'Add some amazing feature'`)**
1. **Push to the branch (`git push origin feature/amazing-feature`)**
1. **Open a Pull Request**, requirements to be merged:
   - All code should pass ruff linting and be formatted with ruff
   - All tests should pass on all supported Python versions
   - Follow existing code style and conventions
