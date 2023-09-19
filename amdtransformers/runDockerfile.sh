sudo rocm-smi --setperflevel high
docker run --rm --mount  src=/media/atek/T7/hfcache,target=/home/workspace/huggingface,type=bind --device /dev/kfd --device /dev/dri/renderD128 amdrocm_transformers_at:1
