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
- base64
Enable base64 mode only where parameter **base64** is set.

### Usage 
#### base64
```bash
POST /document-scanner?base64=true
```
#### jpeg
```bash
POST /document-scanner?output-format=jpeg
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
curl -o result.jpeg -X POST -H "Content-Type: multipart/form-data"  -F "data=@data/854684089.jpg" http://localhost:3000/document-scanner\?output-format\=jpeg
```
or 
```bash
curl  -X POST -H "Content-Type: multipart/form-data"  -F "data=@data/854684089.jpg" http://localhost:3000/document-scanner\?output-format\=jpeg > result.jpeg
```

Open result.jpeg and check the result.
