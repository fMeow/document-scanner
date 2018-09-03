# Restful Document Scanner server on python sanic
- To deploy Restful API document scanner server: docker pull guolilyu/document-scanner
- Stretch based image is a bit ponderous
- alpine based image requires trivial works on building opencv3 and scikit-image. **Help Welcomed**

## Entry and parameters
Only **one** entry is available in this server, that is **/document-scanner**. Only POST method is acceptable.

By default, server accepts image file and return warped image file. Base64 support can be enabled by passing query parameters.

Parameters:
- output-format

   - jpeg
   - jpg
   - png(default)
   
- base64 (default to false)

    Enable base64 mode only when parameter **base64** is set. When in base64 mode, both **input** and **output** are in base64 format.
    
- enhancement (default to false)

    Enhance brightness and contrast, with special care for color enhancement without introducing artifacts. 
    
- grayscale (default to false)

    Grayscale image with enhanced brightness and contrast. 
    
    **This option is only enabled when enhancement option is set.**
    
- id_card (default to false)

    When id_card is set to true, the image is interpreted as optical id card, which is to be rotated and resized to `673x425px`.
    
    Otherwise, the image is regarded as ordinary optical document. In this mode, the image is rotated to portrait when needed, and it's width is limited to `1280px`.


### Usage 
#### base64 png without post processing
```bash
POST /document-scanner?base64=true
```
#### jpeg without post processing
Specify the output format to jpeg. By default, base64 is not enable.
```bash
POST /document-scanner?output-format=jpeg
```
#### png with colorful optical document enhancement
```bash
POST /document-scanner?enhancement=true
```

#### png with grayscale optical document enhancement
```bash
POST /document-scanner?enhancement=true&grayscale=true
```

#### png with colorful optical document enhancement for id card
```bash
POST /document-scanner?enhancement=true&id_card=true
```

## Deployment
This server exposed service on port 3000.

A minimal usage should be like:
```bash
$ docker run -p3000:3000  guolilyu/document-scanner
```

## Minimal and manual E2E test
First locate a jpeg or png file to be scanned and warped on disk, say 854684089.jpg.

Then use *curl* command to POST this image and see what's returned.

```bash
export IMG=1.jpg && curl -o result_$IMG -X POST -H "Content-Type: multipart/form-data"  -F "data=@data/$IMG" http://localhost:3000/document-scanner\?output-format\=jpeg
```
or 
```bash
export IMG=1.jpg && curl  -X POST -H "Content-Type: multipart/form-data"  -F "data=@data/$IMG" http://localhost:3000/document-scanner\?output-format\=jpeg > result_$IMG
```

Open result_1.jpeg and check the result.
