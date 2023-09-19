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
