docker run --rm -it --mount  src=/media/atek/T7/hfcache,target=/home/workspace/huggingface,type=bind --device /dev/kfd --device /dev/dri/renderD128 amdrocm_at_base:latest /bin/bash
