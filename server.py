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
        hsv = cv2.cvtColor(warped, cv2.COLOR_RGB2HSV)

        hue, saturation, gray = cv2.split(hsv)

        """
        Intensity
        """
        sharpen = cv2.GaussianBlur(gray, (5, 5), 3)
        sharpen = cv2.addWeighted(gray, 1.5, sharpen, -0.5, 0)

        # apply adaptive threshold to get black and white effect
        gray = cv2.adaptiveThreshold(sharpen, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 15)
        _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        if request.raw_args.get('grayscale') == 'true':
            result = cv2.GaussianBlur(gray, (5, 5), 3)
        else:
            """
            Saturation enhancement to boost color
            """
            gray_mask = cv2.equalizeHist(gray)
            gray_mask = cv2.GaussianBlur(gray_mask, (5, 5), 3)
            saturation = saturation * (gray_mask < 250)
            saturation = np.uint8(np.clip(saturation * 2, 0, 255))

            result = cv2.cvtColor(cv2.merge((hue, saturation, gray)), cv2.COLOR_HSV2RGB)
            result = cv2.GaussianBlur(result, (5, 5), 3)

    ret, image_stream = cv2.imencode(f".{output_format}", result)

    filename = f"{'.'.join(filename.split('.')[0:-1])}.{output_format}"
    headers = {'Content-length': len(image_stream),
               'Content-Disposition': f'attachment;filename="{urllib.parse.quote(filename)}"'}

    if request.args.get('base64'):
        return text(base64.b16decode(image_stream))
    else:
        return raw(image_stream, headers=headers, content_type=content_type)


if __name__ == "__main__":
    app.go_fast(host="0.0.0.0", port=3000, access_log=True, debug=True)
