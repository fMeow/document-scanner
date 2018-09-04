#!/bin/python3
import cv2
import io
import base64
import urllib.parse

from PIL import Image, ImageEnhance
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
        raise InvalidUsage("")

    # ----------------------------------------
    # read image
    # ----------------------------------------
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

    # ----------------------------------------
    # Reshape and scan
    # ----------------------------------------

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
        warped = (True, intensity.warp(image, scale=resize_ratio))
    else:
        saturation = scanner(hsv[:, :, 1])
        saturation.scan()
        if saturation.corners is not None:
            warped = (True, saturation.warp(image, scale=resize_ratio))
        else:
            warped = (False, image)
    warped_result, warped_image = warped

    # ----------------------------------------
    # Reshape and rotate
    # ----------------------------------------

    height, width, _ = warped_image.shape
    if request.raw_args.get('id_card') == 'true':
        im = Image.fromarray(warped_image)
        if height > width:
            im = im.rotate(90, expand=1)
        im = im.resize((673, 425))
        warped_image = np.array(im)
    else:
        # normal document
        if height < width:
            im = Image.fromarray(warped_image).rotate(-90, expand=1)
            warped_image = np.array(im)

        if width > 1280:
            resize_ratio = width / 1280
            warped_image = cv2.resize(warped_image, (0, 0), interpolation=cv2.INTER_AREA, fx=1 / resize_ratio,
                                      fy=1 / resize_ratio, )

    # ----------------------------------------
    # Enhancement
    # ----------------------------------------
    if request.raw_args.get('enhancement') != 'true':
        result = warped_image
    elif request.raw_args.get('grayscale') == 'true':
        """
        Boost the intensity channel
        """
        # HSV is better than HSI/HLS
        hsv = cv2.cvtColor(warped_image, cv2.COLOR_RGB2HSV)
        hue, saturation, gray = cv2.split(hsv)
        # Sharpen image
        blur = cv2.GaussianBlur(gray, (5, 5), 3)
        sharpen = cv2.addWeighted(gray, 2, blur, -1, 0)
        # a better way to get the effect of histogram equalization
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(sharpen)
        # apply adaptive threshold to get the mask of black items
        # we can infer that darker items may be pixels of interest like text
        gray_mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 15)

        result = gray_mask
    else:
        boosted = Image.fromarray(warped_image)
        boosted = ImageEnhance.Brightness(boosted).enhance(1.45)
        boosted = ImageEnhance.Contrast(boosted).enhance(1.7)
        result = np.array(boosted)

    if request.raw_args.get('id_card') == 'true':
        pass
    else:
        result = cv2.GaussianBlur(result, (3, 3), 5)

    # ----------------------------------------
    # Output
    # ----------------------------------------
    ret, image_stream = cv2.imencode(f".{output_format}", result)

    filename = f"{'.'.join(filename.split('.')[0:-1])}.{output_format}"
    headers = {'Content-length': len(image_stream),
               'Content-Disposition': f'attachment;filename="{urllib.parse.quote(filename)}"',
               'Scanned': warped_result}

    if request.args.get('base64'):
        return text(base64.b16decode(image_stream))
    else:
        return raw(image_stream, headers=headers, content_type=content_type)


if __name__ == "__main__":
    app.go_fast(host="0.0.0.0", port=3000, access_log=True, debug=True)
