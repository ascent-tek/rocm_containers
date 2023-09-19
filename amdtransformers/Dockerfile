# This dockerfile builds an image that contains an installed version of ROCM 5.7.5 from the AMD official amdgpu-install repository
# excluding the dkms driver

FROM amdrocm_at_base:latest

RUN pip3 install transformers
RUN pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm5.6
RUN pip3 install einops accelerate
COPY basictransformer.py /home/workspace

ENTRYPOINT ["python3","/home/workspace/basictransformer.py"]
