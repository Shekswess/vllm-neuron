# GPT-OSS 20B with DFlash on Trn2

This example serves `openai/gpt-oss-20b` with the
[`z-lab/gpt-oss-20b-DFlash`](https://huggingface.co/z-lab/gpt-oss-20b-DFlash)
draft model on one Trn2 instance using TP8.

> [!IMPORTANT]
> The example lives below the existing `mxfp4/` GPT-OSS hierarchy for
> consistency with the repository layout, but DFlash currently requires BF16
> target weights and a BF16 KV cache on Trn2. It does not use MXFP4 and is not
> yet supported on Trn3.

## Supported configuration

- Target: `openai/gpt-oss-20b`
- Drafter: `z-lab/gpt-oss-20b-DFlash`
- Hardware: one Trn2 instance with eight NeuronCores available
- Tensor parallelism: TP8
- Precision: BF16 target, draft, and KV cache
- Speculative tokens: 7 (the checkpoint's proposal block size is 8)
- Scheduling: synchronous
- Prefix caching, chunked prefill, and disaggregated inference: disabled

## Start the server

```bash
./run_tp8.sh
```

The target and draft identifiers can be overridden without editing the script:

```bash
MODEL_ID=/models/gpt-oss-20b \
DRAFT_MODEL_ID=/models/gpt-oss-20b-DFlash \
./run_tp8.sh
```

Cold compilation can take several minutes. Wait for the server to report that
application startup is complete before sending a request.

## Validate

Start with greedy decoding and compare token IDs against the non-speculative
server before measuring performance:

```bash
curl http://localhost:8000/v1/completions \
    -H 'Content-Type: application/json' \
    -d '{
        "model": "openai/gpt-oss-20b",
        "prompt": "Explain why speculative decoding preserves model output.",
        "temperature": 0,
        "max_tokens": 128
    }'
```

Real-hardware validation should cover exact greedy token equality, acceptance
length, target-model call reduction, TTFT, TPOT, throughput, and HBM usage.
