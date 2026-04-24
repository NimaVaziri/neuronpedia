#!/bin/sh
set -eu

{
  echo "Starting Qwen3-4B IA+PC full eval at $(date -Is)"
  echo "Canonical eval runner=apps/graph/eval/eval_runner.py"
  echo

  cd /root/ce-qweb3-4b/apps/graph
  export MODEL_ID=Qwen/Qwen3-4B
  export STEER_URL=http://127.0.0.1:5004
  export STEER_SECRET=SECRET
  export EVAL_GRAPH_DIR=/root/ce-qweb3-4b/apps/graph/eval/models/qwen3-4b/graphs
  echo "MODEL_ID=$MODEL_ID"
  echo "STEER_URL=$STEER_URL"
  echo "EVAL_GRAPH_DIR=$EVAL_GRAPH_DIR"
  echo

  python3 eval/eval_runner.py --method ia_pc --prompts all
} >> /tmp/neuronpedia-qwen-iapc-eval.log 2>&1
