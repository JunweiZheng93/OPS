# the docker image is based on Ubuntu18.04, CUDA11.4.1 and cdDNN8.2.4
FROM nvidia/cudagl:11.4.1-devel-ubuntu18.04
SHELL ["/bin/bash", "-c"]

# install some necessary packages
ENV DEBIAN_FRONTEND=noninteractive
RUN apt update && apt install -y git wget vim doxygen curl libjsoncpp-dev libepoxy-dev libglm-dev libosmesa6 libosmesa6-dev libglew-dev libopencv-dev python3-opencv python3-distutils openssh-server

# uninstall the default python3.6
RUN apt remove python3 -y && apt autoremove -y

# install python3.10
RUN apt install -y build-essential checkinstall libreadline-gplv2-dev libncursesw5-dev libssl-dev libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev libffi-dev zlib1g-dev && \
    wget https://www.python.org/ftp/python/3.10.9/Python-3.10.9.tgz && \
    tar xzf Python-3.10.9.tgz && \
    cd Python-3.10.9 && \
    ./configure --enable-optimizations && \
    make install && \
    cd / && \
    rm -rf Python-3.10.9*

# set python and pip instead of python3 and pip3
RUN update-alternatives --install /usr/bin/python python /usr/local/bin/python3 1
RUN update-alternatives --install /usr/bin/pip pip /usr/local/bin/pip3 1

# install python packages
RUN pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118 && \
    pip install lit==18.1.8 numpy==1.23.1 cmake==3.30.4 openmim==0.3.9 timm==0.9.8 einops==0.7.0 ftfy==6.1.1 pkbar==0.5 prettytable==3.9.0 py360convert==0.1.0 regex==2023.10.3 six==1.16.0 && \
    mim install mmengine==0.9.0 mmcv==2.1.0 mmsegmentation==1.2.2

# install cuDNN
RUN wget https://developer.download.nvidia.com/compute/redist/cudnn/v8.2.4/cudnn-11.4-linux-x64-v8.2.4.15.tgz && \
    tar -xzvf cudnn-11.4-linux-x64-v8.2.4.15.tgz && \
    cp cuda/include/cudnn.h /usr/local/cuda/include && \
    cp cuda/lib64/libcudnn* /usr/local/cuda/lib64 && \
    chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn* && \
    rm -rf cuda/ && \
    rm -rf cudnn-11.4-linux-x64-v8.2.4.15.tgz

# install gcc9
RUN apt install -y software-properties-common && add-apt-repository -y ppa:ubuntu-toolchain-r/test
RUN apt install -y gcc-9 g++-9
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 60 --slave /usr/bin/g++ g++ /usr/bin/g++-9

# setup ssh for the development with IDE
RUN echo "PermitRootLogin yes" >> /etc/ssh/sshd_config && echo "service ssh start" >> /root/.bashrc
