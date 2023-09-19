# This dockerfile builds an image that contains an installed version of ROCM 5.7.5 from the AMD official amdgpu-install repository
# excluding the dkms driver

FROM amdrocm_at_base:latest


# IMPORTANT, CUSTOM PYTORCH MUST BE BUILT AND INSTALLED BEFORE DEEPSPEED IS INSTALLED!

# Install prerequisites for pytorch build.
RUN apt-get install libopenmpi-dev git -y

# Download and rebuild Pytorch in a custom manner. 
RUN git clone https://github.com/pytorch/pytorch.git; cd pytorch; git checkout heads/main-0-gce608712cbb ;git submodule update --init --recursive
COPY pytorchbuild.sh /home/workspace/pytorch/pytorchbuild.sh
RUN apt install cmake -y 
RUN amdgpu-install --no-dkms --usecase=rocmdev,openclsdk,hiplibsdk,openmpsdk,mlsdk -y
RUN cd pytorch;./pytorchbuild.sh

# Download and custom build Deepspeed for AMD
# This fork contains a few changes that are required for DS to build with rocm.
RUN apt install libaio-dev -y
RUN git clone https://github.com/peterjweir/DeepSpeed.git; cd DeepSpeed;git checkout peterbuild; git submodule update --init --recursive
#RUN git clone https://github.com/microsoft/DeepSpeed.git; cd DeepSpeed; git submodule update --init --recursive
COPY deepspeedbuild.sh /home/workspace/DeepSpeed/deepspeedbuild.sh
# Patch Rocm headers to resolve RocBlas definition issue. 
RUN sed -i '/#include <stdint.h>/a #include <rocblas/rocblas.h>' /opt/rocm/include/hipblas/hipblas.h
RUN cd DeepSpeed;./deepspeedbuild.sh


# Download the standard DeepSpeed Example for baseline performance
RUN git clone https://github.com/microsoft/DeepSpeedExamples.git

RUN git clone https://github.com/huggingface/transformers-bloom-inference.git

RUN pip3 install transformers einops accelerate

COPY runDemonstration.sh /home/workspace

ENTRYPOINT ["bash","/home/workspace/runDemonstration.sh"]
