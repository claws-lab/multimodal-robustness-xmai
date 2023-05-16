# Referencing https://github.com/facebookresearch/detectron2/blob/main/docker/Dockerfile
FROM nvidia/cuda:10.2-cudnn8-devel-ubuntu18.04

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y \
	python3-opencv ca-certificates python3-dev git wget sudo  \
	cmake ninja-build protobuf-compiler libprotobuf-dev build-essential && \
    rm -rf /var/lib/apt/lists/*
RUN ln -sv /usr/bin/python3 /usr/bin/python

# create a non-root user
ARG USER_ID=1000
RUN useradd -m --no-log-init --system  --uid ${USER_ID} appuser -g sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER appuser
WORKDIR /home/appuser

ENV PATH="/home/appuser/.local/bin:${PATH}"
RUN wget https://bootstrap.pypa.io/pip/3.6/get-pip.py && \
	python3 get-pip.py --user && \
	rm get-pip.py

RUN pip install --user torch==1.10 torchvision==0.11.0 -f https://download.pytorch.org/whl/cu102/torch_stable.html

# clone bottom-up-attention repo
RUN git clone --recursive https://github.com/MILVLG/bottom-up-attention.pytorch
# set FORCE_CUDA because during `docker build` cuda is not accessible
ENV FORCE_CUDA="1"

ARG TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"
ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"

RUN pip install --user -e bottom-up-attention.pytorch/detectron2

RUN pip install --user packaging
RUN cd bottom-up-attention.pytorch && \
    git clone https://github.com/NVIDIA/apex.git && \
    cd apex && \
    python setup.py install --user && \
    cd .. && \
    python setup.py build develop --user
RUN pip install --user ray jupyter pandas scikit-image

# download model
RUN wget -O /home/appuser/bottom-up-attention.pytorch/bua-caffe-frcn-r152.pth \
    https://awma1-my.sharepoint.com/:u:/g/personal/yuz_l0_tn/ETDgy4bY0xpGgsu5tEMzgLcBQjAwpnkKkltNTtPVuMj4GQ?download=1