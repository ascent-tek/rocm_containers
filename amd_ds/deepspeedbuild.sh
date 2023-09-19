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
