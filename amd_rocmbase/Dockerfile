# This dockerfile builds an image that contains an installed version of ROCM 5.7.5 from the AMD official amdgpu-install repository
# excluding the dkms driver

FROM ubuntu:22.04

RUN apt-get update; apt-get install -y libstdc++-12-dev wget
RUN wget https://repo.radeon.com/amdgpu-install/5.6/ubuntu/jammy/amdgpu-install_5.6.50600-1_all.deb; apt install -y ./amdgpu-install*.deb
#RUN wget https://repo.radeon.com/amdgpu-install/5.7/ubuntu/jammy/amdgpu-install_5.7.50700-1_all.deb; apt install -y ./amdgpu-install*.deb

RUN amdgpu-install --no-dkms --usecase=rocmdev,openclsdk,hiplibsdk,openmpsdk,mlsdk -y

RUN rocm-smi

RUN apt-get install python3-pip -y

RUN mkdir -p /home/workspace
RUN mkdir -p /home/workspace/huggingface

WORKDIR /home/workspace
ENV HF_HOME=/home/workspace/huggingface
