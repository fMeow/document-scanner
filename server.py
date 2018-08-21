#!/bin/python3
import cv2
import io
import base64
import urllib.parse

from sanic import Sanic
from sanic_compress import Compress
from sanic.response import raw, text
from sanic.exceptions import InvalidUsage, ServerError

import numpy as np
from doc_scanner import scanner

app = Sanic()
Compress(app)


@app.post('/document-scanner')
async def document_scanner(request):
    output_format = request.args.get('output-format')
    if output_format is None:
        output_format = 'png'
    if output_format in ['jpg', 'jpeg']:
        content_type = f'image/jpeg'
    elif output_format in ['png']:
        content_type = f'image/png'
    else:
        raise ServerError("")

    if request.raw_args.get('base64') == 'true':
        raw_image = base64.b64decode(request.body)
    else:
        if len(request.files) != 1:
            raise InvalidUsage("Only accept one file at once")
        _, value = request.files.popitem()
        raw_image = value[0].body
        filename = value[0].name
    image_stream = io.BytesIO(raw_image)
    image_stream.seek(0)
    file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None:
        return InvalidUsage("")
    height, width, _ = image.shape
    if height > width:
        resize_ratio = width / 500
    else:
        resize_ratio = height / 500
    resized_image = cv2.resize(image, (0, 0), interpolation=cv2.INTER_AREA,
                               fx=1 / resize_ratio, fy=1 / resize_ratio, )

    # Convert RGB to HSV colorspace
    hsv = cv2.cvtColor(resized_image, cv2.COLOR_RGB2HSV)

    """
    First scan gray scale component of images and if failed, then turn to saturation images.
    """
    # hue ranges from 0-180
    # hue = scanner(hsv[:, :, 0])
    intensity = scanner(hsv[:, :, 2])
    intensity.scan()
    if intensity.corners is not None:
        # Find corners in intensity images
        warped = intensity.warp(image, scale=resize_ratio)
    else:
        saturation = scanner(hsv[:, :, 1])
        saturation.scan()
        if saturation.corners is not None:
            warped = saturation.warp(image, scale=resize_ratio)
        else:
            warped = None

    if warped is None:
        raise ServerError("Failed to find boundary")

    if request.raw_args.get('enhancement') != 'true':
        result = warped
    else:
        # convert the warped image to grayscale
        # hls = cv2.cvtColor(warped, cv2.COLOR_RGB2HLS)
        # hue, gray, saturation = cv2.split(hls)

        # HSV is better than HSI/HLS
        hsv = cv2.cvtColor(warped, cv2.COLOR_RGB2HSV)
        hue, saturation, gray = cv2.split(hsv)

        """
        Boost the intensity channel
        """
        # Sharpen image
        blur = cv2.GaussianBlur(gray, (5, 5), 3)
        sharpen = cv2.addWeighted(gray, 2, blur, -1, 0)
        # a better way to get the effect of histogram equalization
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(sharpen)
        # apply adaptive threshold to get the mask of black items
        # we can infer that darker items may be pixels of interest like text
        gray_mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 15)

        # filter by morphological opening to eliminate salt noise
        kernel_size = 3
        kernel = np.ones((kernel_size, kernel_size), dtype=np.int8)
        gray_mask = cv2.morphologyEx(gray_mask, cv2.MORPH_OPEN, kernel)

        # boost brightness with special care of overflow
        value = 150
        gray = np.where((255 - gray) > value, gray + value, 255)
        # darken pixels of interest
        gray[gray_mask == 255] = np.uint8(gray[gray_mask == 255] * 0.3)

        if request.raw_args.get('grayscale') == 'true':
            result = gray
        else:
            result = cv2.cvtColor(cv2.merge((hue, saturation, gray,)), cv2.COLOR_HSV2RGB)

        result = cv2.GaussianBlur(result, (5, 5), 3)

    ret, image_stream = cv2.imencode(f".{output_format}", result)

    filename = f"{'.'.join(filename.split('.')[0:-1])}.{output_format}"
    headers = {'Content-length': len(image_stream),
               'Content-Disposition': f'attachment;filename="{urllib.parse.quote(filename)}"'}

    if request.args.get('base64'):
        return text(base64.b16decode(image_stream))
    else:
        return raw(image_stream, headers=headers, content_type=content_type)


def homomorphic_filter(y, rh=2.5, rl=0.5, cutoff=32):
    rows, cols = y.shape
    y_log = np.log(y + 0.01)

    y_fft = np.fft.fft2(y_log)

    y_fft_shift = np.fft.fftshift(y_fft)

    DX = cols / cutoff
    G = np.ones((rows, cols))
    for i in range(rows):
        for j in range(cols):
            G[i][j] = ((rh - rl) * (1 - np.exp(-((i - rows / 2) ** 2 + (j - cols / 2) ** 2) / (2 * DX ** 2)))) + rl

    result_filter = G * y_fft_shift

    result_interm = np.real(np.fft.ifft2(np.fft.ifftshift(result_filter)))

    return np.exp(result_interm)


if __name__ == "__main__":
    app.go_fast(host="0.0.0.0", port=3000, access_log=True, debug=True)
