FROM python:3.7-stretch

WORKDIR /usr/src/app

ADD requirement.txt requirement.txt
RUN pip install numpy cython --no-cache-dir -i https://mirrors.aliyun.com/pypi/simple/ && \
pip install -r requirement.txt --no-cache-dir -i https://mirrors.aliyun.com/pypi/simple/

RUN pip install git+https://github.com/guoli-lyu/document-scanner.git@c044eb4

ADD server.py server.py

CMD python3 -m sanic server.app --host=0.0.0.0 --port=3000
