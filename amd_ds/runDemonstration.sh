#!/bin/bash
#This will call the pertinent examples with arguments from the upstream repositories

echo "Single GPU DS inference, 'facebook/opt-6.7b'"

deepspeed --num_gpus 1 --num_nodes 1  DeepSpeedExamples/inference/huggingface/text-generation/inference-test.py --model "facebook/opt-6.7b" --batch_size 20 --test_performance

echo "Non Offload baseline (6.7b) (notice this is slower than straight inference due to library in use)"

cd transformers-bloom-inference/bloom-inference-scripts

deepspeed --num_gpus 1 bloom-ds-zero-inference.py --name "facebook/opt-6.7b" --benchmark --batch_size=20
echo "CPU Offload Comparison (6.7b)"

deepspeed --num_gpus 1 bloom-ds-zero-inference.py --cpu_offload --name "facebook/opt-6.7b" --benchmark --batch_size=20


echo "CPU Offload Comparison (13b, cannot run without offload or quantization)"

deepspeed --num_gpus 1 bloom-ds-zero-inference.py --cpu_offload --name "facebook/opt-13b" --benchmark --batch_size=20
echo "NVME offload requires directory to test, but the command is documented below"

echo "In order to run you must add a volume mount point to /home/workspace/aioffloaddir from a fast storage device on your host"

deepspeed --num_gpus 1 bloom-ds-zero-inference.py --nvme_offload_path /home/workspace/aioffloaddir --name "facebook/opt-30b" --benchmark --batch_size=20

