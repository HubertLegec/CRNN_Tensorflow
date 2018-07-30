FROM hubertlegec/opencv-python:1.0

MAINTAINER Hubert Legęc <hubert.legec@gmail.com>

COPY requirements.txt /requirements.txt
RUN pip install -r requirements.txt

RUN mkdir /data
COPY /data/ /data

RUN mkdir /app
COPY /src/ /app
WORKDIR /app

RUN export PYTHONPATH=$PYTHONPATH:/app

ENTRYPOINT python src/test.py --dataset_dir /data/ --weights_path model/shadownet/shadownet_2017-09-29-19-16-33.ckpt-39999