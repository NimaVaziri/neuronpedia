#!/bin/sh
set -eu

cd /root/ce-qweb3-4b/apps/graph
export PREBUILD_SCORER_ON_GENERATE=false
echo "Starting Neuronpedia graph server at $(date -Is)" > /tmp/neuronpedia-graph.log
exec poetry run python start.py \
  --model_id Qwen/Qwen3-4B \
  --transcoder_set mwhanna/qwen3-4b-transcoders \
  --device cuda \
  --model_dtype bfloat16 \
  >> /tmp/neuronpedia-graph.log 2>&1
