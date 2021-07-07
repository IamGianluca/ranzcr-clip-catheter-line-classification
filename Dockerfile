FROM nvcr.io/nvidia/pytorch:21.04-py3  

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y

# install extra python packages
RUN pip install albumentations timm pytorch-lightning opencv-python
RUN pip install dvc --ignore-installed ruamel-yaml

# install personal library
COPY src /src
RUN cd /src && pip install -e . 

# ENTRYPOINT ["sh", "run_3090.sh"]