# AMD Radeon RX7900XTX running DeepSpeed

<img style="float: left;" src="rx7900xtx.jpg" width=50%><img style="float: right;" src="deepspeed.png" width=50%>

&nbsp;



<font size="4">This guide walks through what was done to get the AMD Radeon RX7900XTX to run inference with offload support using DeepSpeed.</font>

| [Introduction](#introduction) | [Hardware](#hardware-configuration) | [Software](#software-configuration) | [Running](#run-a-model-with-deepspeed) | [Conclusion](#conclusion) |
## Introduction
Democratizing AI requires all GPU manufactures to be able to run open-source AI models.  AMD is expanding their focus toward AI enablement. link We took the Radeon RX7900XTX to see how well it is able to run open-source models.

## Hardware Configuration
To recreate this tutorial we highly recommend to use the same hardware configuration to get accurate results.  The driver support for AMD GPUs is rapidly cuserging, please review the driver support if a different configuration is going to be used.

<b>Motherboard</b>: [SuperMicro M12SWA-TF](https://www.supermicro.com/en/products/motherboard/m12swa-tf)

<b>CPU</b>: AMD Ryzen Threadripper Pro 5955W

<b>GPU</b>: Radeon RTX7900XTX

<b>RAM</b>: 256 DDr4 Ecc RDimm


## Software Configuration
### Install the latest mainline kernel
Ubuntu 22.04 LTS was the initial OS installation.  The Kernel was then updated to 6.5.2 to include the latest kernel fixes for the Radeon card (with the 6.2.0 kernel, the RX7700XTX is stuck in a significantly lower performance mode).

```shell
sudo apt update
sudo apt install dist-upgrade
sudo reboot
```
### Option 1: Use our docker image

### Option 2: Install everything yourself
Install ROCm5.6.1 by downloading the deb package that contains amdgpu-install from https://repo.radeon.com/amdgpu-install/5.6.1/ubuntu/jammy/
```shell
wget https://repo.radeon.com/amdgpu-install/5.6.1/ubuntu/jammy/amdgpu-install_5.6.50601-1_all.deb
sudo apt install ./amdgpu-install_5.6.50601-1_all.deb 
```
<b>amdgpu-install</b> will install ROCm.  (The amdgpu driver is already in the latest linux kernel, so you can skip dealing with it using the --no-dkms option)
<br/>
Afterward, you should find ROCm installed in <b>/opt/rocm</b>

```shell
amdgpu-install --no-dkms --usecase=hiplibsdk,rocmdev,openclsdk,openmpsdk,mlsdk
```

You'll need to build PyTorch yourself to make sure RCCL is enabled, we included the specific SHA we tested.

(Optional) You'll be installing your built pytorch after you build it, so it's probably a good idea to work in a python virtual environment
```shell
sudo apt install python3.10-venv
python3 -m venv venv_amd
source ./venv_amd/bin/activate 
```

```shell
sudo apt install libopenmpi-dev git -y
git clone https://github.com/pytorch/pytorch.git
cd pytorch
git checkout heads/main-0-gce608712cbb
git submodule update --init --recursive
```

Here are the options and commands we use to build and install pytorch, it will take a bit of time to complete.
```shell
export USE_NCCL=1
export USE_QNNPACK=0
export USE_CUDA=OFF
export USE_RCCL=ON
export USE_ROCM=ON
export USE_PYTORCH_QNNPACK=0
export TORCH_CUDA_ARCH_LIST="7.0;7.2;8.6;8.7;8.9;9.0"
export PYTORCH_ROCM_ARCH="gfx1100"
export MIOPEN_INCLUDE_DIR="/usr/include/x86_64-linux-gnu"
export MIOPEN_LIBRARY="/usr/lib/x86_64-linux-gnu"
export PYTORCH_BUILD_NUMBER=1

pip3 install -r requirements.txt
pip3 install Cython
pip3 install scikit-build
pip3 install ninja

python3 tools/amd_build/build_amd.py
python3 setup.py bdist_wheel --cmake

pip3 install --force-reinstall --no-cache dist/torch*.whl
```

DeepSpeed will also need to be built for ROCm support.  Some small workarounds were needed to get it to build, and a hack to ROCm headers to resolve some dependency issues in DeepSpeed's CUDA Ops was needed.  Take a look at our fork if you are interested in more details.

Grab our fork of DeepSpeed
```shell
sudo apt install libaio-dev -y
git clone https://github.com/peterjweir/DeepSpeed.git 
cd DeepSpeed
git checkout peterbuild
git submodule update --init --recursive
```

This is the quick hack to ROCm that resolves missing typedefs.
```shell
sudo sed -i '/#include <stdint.h>/a #include <rocblas/rocblas.h>' /opt/rocm/include/hipblas/hipblas.h
```

Here are the options and commands to build DeepSpeed.  A lot of the DeepSpeed Ops are disabled, we only enabled transformer inference.
```shell
export DS_BUILD_OPS=1
export DS_BUILD_SPARSE_ATTN=0
export DS_BUILD_QUANTIZER=0
export DS_BUILD_RANDOM_LTD=0
export DS_BUILD_TRANSFORMER_INFERENCE=1
export DS_BUILD_CPU_ADAGRAD=0
export DS_BUILD_CPU_ADAM=0
export DS_BUILD_TRANSFORMER=0
export DS_BUILD_STOCHASTIC_TRANSFORMER=0 

pip3 install -r requirements/requirements-inf.txt
pip3 install Cython
pip3 install scikit-build
pip3 install ninja

python3 setup.py bdist_wheel

pip3 install --no-cache dist/deepspeed*.whl
```


## Run a model with DeepSpeed
Before you run a model, but after ROCm is installed, you'll need to use rocm-smi to set the card to high power mode.
```shell
rocm-smi --setperflevel high
```

To run DeepSpeed, we just grab the <b>DeepSpeedExamples</b> and <b>transformers-bloom-inference</b> repos for examples to run and compare (and install their dependencies).

```shell
git clone https://github.com/microsoft/DeepSpeedExamples.git
git clone https://github.com/huggingface/transformers-bloom-inference.git
pip3 install transformers einops accelerate
```

Below is the run script we've included, with examples of different ways to invoke DeepSpeed for the various types of offload you are looking for.
```shell
#!/bin/bash
#This will call the pertinent examples with arguments from the upstream repositories

echo "Single GPU DS inference, 'facebook/opt-6.7b'"
deepspeed --num_gpus 1 --num_nodes 1  DeepSpeedExamples/inference/huggingface/text-generation/inference-test.py --model "facebook/opt-6.7b" --batch_size 20 --test_performance

echo "Non Offload baseline (6.7b) (notice this is slower than straight inference due to library in use)"
deepspeed --num_gpus 1 transformers-bloom-inference/bloom-inference-scripts/bloom-ds-zero-inference.py --name "facebook/opt-6.7b" --benchmark

echo "CPU Offload Comparison (6.7b)"
deepspeed --num_gpus 1 transformers-bloom-inference/bloom-inference-scripts/bloom-ds-zero-inference.py --cpu_offload --name "facebook/opt-6.7b" --benchmark


echo "CPU Offload Comparison (13b, cannot run without offload or quantization)"
deepspeed --num_gpus 1 transformers-bloom-inference/bloom-inference-scripts/bloom-ds-zero-inference.py --cpu_offload --name "facebook/opt-13b" --benchmark

echo "NVME offload requires directory to test, but the command is documented below"
mkdir nvmeoffload
deepspeed --num_gpus 1 transformers-bloom-inference/bloom-inference-scripts/bloom-ds-zero-inference.py --nvme_offload_path ./nvmeoffload --name \"facebook/opt-13b\" --benchmark
```



## Conclusion
With DeepSpeed you can mess around with much larger models than your GPU hardware naturally supports.  Also in our testing, the offload bottleneck dwarfs any hardware differences, making the AMD RX7900XTX perform pretty much just as well as a higher end Nvidia card.  