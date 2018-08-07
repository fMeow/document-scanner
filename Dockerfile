FROM python:3.7-stretch

WORKDIR /usr/src/app

ADD requirement.txt requirement.txt
RUN pip install numpy cython --no-cache-dir -i https://mirrors.ustc.edu.cn/pypi/web/simple && \
pip install -r requirement.txt --no-cache-dir -i https://mirrors.ustc.edu.cn/pypi/web/simple

ADD server-requirement.txt requirement.txt
RUN pip install -r requirement.txt --no-cache-dir -i https://mirrors.ustc.edu.cn/pypi/web/simple

ADD . .

CMD python3 -m sanic server.app --host=0.0.0.0 --port=3000
