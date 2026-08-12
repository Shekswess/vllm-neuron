#!/bin/bash
# GPT-OSS 20B with DFlash on one Trn2 instance using TP8 and BF16.
#
# This script intentionally disables unsupported cache and scheduling modes.
# Despite its location below the existing mxfp4/ examples hierarchy, the
# current DFlash preview does not support MXFP4 or Trn3.

set -euo pipefail
set -x

MODEL_ID="${MODEL_ID:-openai/gpt-oss-20b}"
DRAFT_MODEL_ID="${DRAFT_MODEL_ID:-z-lab/gpt-oss-20b-DFlash}"

export VLLM_NEURON_COMPILATION_TIMEOUT="${VLLM_NEURON_COMPILATION_TIMEOUT:-1200}"
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-1200}"

echo "Starting GPT-OSS 20B with DFlash: target=$MODEL_ID draft=$DRAFT_MODEL_ID"
echo "  Hardware: Trn2, precision: BF16, port: 8000, TP8, speculative tokens: 7"

vllm serve "$MODEL_ID" \
    --tensor-parallel-size 8 \
    --dtype bfloat16 \
    --max-model-len 16384 \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 4 \
    --no-async-scheduling \
    --no-enable-chunked-prefill \
    --no-enable-prefix-caching \
    --hf-overrides '{"quantization_config": {}}' \
    --speculative-config "{\"method\":\"dflash\",\"model\":\"${DRAFT_MODEL_ID}\",\"num_speculative_tokens\":7}" \
    --additional-config '{
        "neuron_config": {
            "quantization": "bf16",
            "kv_segment_size_buckets": [8192],
            "num_batched_tokens_buckets": [8192],
            "num_seqs_buckets": [4]
        }
    }'
