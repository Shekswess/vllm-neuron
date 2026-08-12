# SPDX-License-Identifier: Apache-2.0
"""DFlash parallel draft proposer for the Neuron model runner."""

import contextlib
import time

import torch

from vllm_neuron.compile.backend import model_forward_context
from vllm_neuron.compile.platform import get_platform_target
from vllm_neuron.functional.attention.attention_decode_mask import (
    gen_attention_decode_mask,
)
from vllm_neuron.metrics import COMPILATION_TIME, NEFF_EXECUTION_COUNT
from vllm_neuron.model.llama3.eagle3_model import compute_slot_mapping

from .eagle import EagleProposer


def dflash_target_capture_layer_ids(hf_config) -> tuple[int, ...] | None:
    """Translate DFlash decoder-output IDs to Neuron residual boundaries.

    The z-lab checkpoint IDs are zero-based decoder layer outputs. Neuron's
    target model captures the residual stream immediately before a layer, so
    output of decoder layer ``i`` is the boundary before layer ``i + 1``.
    """
    dflash_config = getattr(hf_config, "dflash_config", None)
    if not isinstance(dflash_config, dict):
        return None
    layer_ids = dflash_config.get("target_layer_ids")
    if not layer_ids or not isinstance(layer_ids, (list, tuple)):
        return None
    return tuple(int(layer_id) + 1 for layer_id in layer_ids)


class DFlashProposer(EagleProposer):
    """Generate all DFlash proposal tokens in one non-causal draft pass."""

    expected_method = "dflash"

    def __init__(self, vllm_config, device, on_device_sampling=True):
        super().__init__(vllm_config, device, on_device_sampling)
        if not on_device_sampling:
            raise ValueError("DFlash on Neuron requires on-device sampling")
        if vllm_config.scheduler_config.async_scheduling:
            raise ValueError(
                "DFlash on Neuron initially supports synchronous scheduling only; "
                "pass --no-async-scheduling"
            )
        if vllm_config.scheduler_config.enable_chunked_prefill:
            raise ValueError(
                "DFlash on Neuron does not yet support chunked prefill; "
                "pass --no-enable-chunked-prefill"
            )
        if vllm_config.kv_transfer_config is not None:
            raise ValueError(
                "DFlash on Neuron does not yet support disaggregated inference"
            )
        if vllm_config.cache_config.enable_prefix_caching:
            raise ValueError("DFlash on Neuron does not yet support prefix caching")
        if self.num_speculative_tokens != 7:
            raise ValueError(
                "z-lab/gpt-oss-20b-DFlash requires num_speculative_tokens=7"
            )
        if vllm_config.parallel_config.tensor_parallel_size != 8:
            raise ValueError(
                "Initial DFlash support is validated for tensor_parallel_size=8"
            )
        platform = get_platform_target()
        if platform != "trn2":
            raise ValueError(
                "Initial DFlash support is validated only on Trn2; "
                f"detected platform {platform!r}"
            )
        target_quantization = vllm_config.additional_config.get(
            "neuron_config", {}
        ).get("quantization")
        if target_quantization not in (None, "bf16"):
            raise ValueError(
                "Initial Trn2 DFlash support requires quantization='bf16', "
                f"got {target_quantization!r}"
            )
        target_config = vllm_config.model_config.hf_text_config
        target_type = target_config.model_type
        if target_type != "gpt_oss":
            raise ValueError(
                f"Initial DFlash support requires a GPT-OSS target, got {target_type!r}"
            )
        target_geometry = (
            target_config.hidden_size,
            target_config.num_hidden_layers,
            target_config.vocab_size,
        )
        if target_geometry != (2880, 24, 201088):
            raise ValueError(
                "z-lab/gpt-oss-20b-DFlash is trained only for the GPT-OSS 20B "
                f"target; got geometry {target_geometry}"
            )
        if vllm_config.model_config.dtype != torch.bfloat16:
            raise ValueError(
                "Initial Trn2 DFlash support requires target dtype=bfloat16"
            )
        draft_config = self.draft_model_config.hf_config
        if draft_config.block_size != self.num_speculative_tokens + 1:
            raise ValueError(
                "DFlash proposal geometry mismatch: checkpoint block_size must equal "
                "num_speculative_tokens + 1"
            )
        dflash_config = draft_config.dflash_config
        if dflash_config.get("mask_token_id") != 200000 or dflash_config.get(
            "target_layer_ids"
        ) != [1, 6, 11, 16, 21]:
            raise ValueError(
                "Unsupported DFlash checkpoint metadata: expected mask_token_id=200000 "
                "and target_layer_ids=[1, 6, 11, 16, 21]"
            )
        if vllm_config.cache_config.cache_dtype not in ("auto", "bfloat16"):
            raise ValueError("Initial DFlash support requires a BF16 KV cache")

    def _build_synthetic_inputs(self, num_tokens, num_reqs, device=None):
        assert self.model is not None
        if device is None:
            device = self.device
        hidden_size = self.model.config.hidden_size
        num_features = len(self.model.target_layer_ids)
        tokens_per_req = num_tokens // num_reqs
        return {
            "target_token_ids": torch.ones(
                num_tokens, dtype=torch.int32, device=device
            ),
            "target_positions": torch.arange(
                tokens_per_req, dtype=torch.long, device=device
            ).repeat(num_reqs),
            "target_hidden_states": torch.ones(
                num_tokens,
                hidden_size * num_features,
                dtype=torch.bfloat16,
                device=device,
            ),
            "last_token_indices": torch.tensor(
                [(i + 1) * tokens_per_req - 1 for i in range(num_reqs)],
                dtype=torch.long,
                device=device,
            ),
            "raw_sampled_token_ids": torch.ones(
                num_reqs, 1, dtype=torch.int32, device=device
            ),
        }

    def _valid_sample_count(self, raw_sampled_token_ids: torch.Tensor) -> torch.Tensor:
        valid = (raw_sampled_token_ids >= 0) & (
            raw_sampled_token_ids < self.model.config.vocab_size
        )
        return valid.to(torch.int32).sum(dim=1)

    def _last_valid_sample(self, raw_sampled_token_ids: torch.Tensor) -> torch.Tensor:
        valid_count = self._valid_sample_count(raw_sampled_token_ids)
        last = valid_count.sub(1).clamp_min(0)
        return raw_sampled_token_ids.gather(1, last.unsqueeze(1)).squeeze(1)

    def _last_valid_context_indices(
        self,
        last_token_indices: torch.Tensor,
        raw_sampled_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Move each context boundary back over rejected verification tokens."""
        if raw_sampled_token_ids.shape[1] == 1:
            return last_token_indices
        valid_count = self._valid_sample_count(raw_sampled_token_ids)
        num_rejected = (self.num_speculative_tokens + 1 - valid_count).clamp_min(0)
        return last_token_indices - num_rejected.to(last_token_indices.dtype)

    def propose(
        self,
        target_token_ids,
        target_positions,
        target_hidden_states,
        last_token_indices,
        attn_metadata,
        raw_sampled_token_ids,
        prev_sampled_token_ids=None,
        prev_num_draft_tokens=None,
        req_indices_per_token=None,
        is_warmup=False,
        model_override=None,
    ):
        del target_token_ids, prev_sampled_token_ids, prev_num_draft_tokens
        del req_indices_per_token
        assert self.model is not None
        model = model_override if model_override is not None else self.model
        target_device = (
            torch.device("meta")
            if target_positions.device.type == "meta"
            else self.device
        )

        target_positions = target_positions.to(target_device)
        target_hidden_states = target_hidden_states.to(target_device)
        last_token_indices = last_token_indices.to(target_device)
        raw_sampled_token_ids = raw_sampled_token_ids.to(target_device)
        batch_size = last_token_indices.shape[0]
        query_len = self.num_speculative_tokens + 1

        first_layer_name = self.attn_layer_names[0]
        first_meta = attn_metadata[first_layer_name]
        block_table = first_meta["block_table_tensor"].to(target_device)
        block_size = first_meta["block_size"]
        context_slot_mapping = first_meta["slot_mapping"].to(target_device)
        max_slot = self.model.layers[0].self_attn.k_cache.shape[0] * block_size
        context_slot_mapping = torch.where(
            (context_slot_mapping < 0) | (context_slot_mapping >= max_slot),
            torch.zeros_like(context_slot_mapping),
            context_slot_mapping,
        )

        bonus = self._last_valid_sample(raw_sampled_token_ids)
        last_token_indices = self._last_valid_context_indices(
            last_token_indices, raw_sampled_token_ids
        )
        input_ids = torch.full(
            (batch_size, query_len),
            self.model.mask_token_id,
            dtype=torch.int32,
            device=target_device,
        )
        input_ids[:, 0] = bonus
        base_positions = target_positions[last_token_indices] + 1
        offsets = torch.arange(
            query_len, device=target_device, dtype=base_positions.dtype
        )
        query_positions = base_positions[:, None] + offsets[None, :]
        query_slot_mapping = compute_slot_mapping(
            query_positions.reshape(-1), block_table, block_size
        )

        q_heads = self.model.layers[0].self_attn.num_attention_heads_per_rank
        s_prior = block_table.shape[1] * block_size
        non_causal_active = torch.ones(
            query_len,
            batch_size,
            q_heads,
            query_len,
            dtype=torch.float32,
            device=target_device,
        )
        attention_mask = gen_attention_decode_mask(
            pos_ids=query_positions.reshape(1, -1).to(torch.float32),
            bs=batch_size,
            q_head=q_heads,
            s_active=query_len,
            s_prior=s_prior,
            block_len=block_size,
            active_mask=non_causal_active,
        )

        draft_metadata = {name: attn_metadata[name] for name in self.attn_layer_names}
        start = time.perf_counter()
        with (
            contextlib.nullcontext()
            if is_warmup
            else model_forward_context(self.vllm_config)
        ):
            draft_token_ids = model(
                input_ids=input_ids.reshape(-1),
                positions=query_positions.reshape(-1),
                target_hidden_states=target_hidden_states,
                context_positions=target_positions,
                context_slot_mapping=context_slot_mapping,
                query_slot_mapping=query_slot_mapping,
                attn_metadata=draft_metadata,
                active_mask=attention_mask,
                rank=self.rank_tensor.to(target_device),
            )
        elapsed = time.perf_counter() - start
        bucket_name = f"dflash_b{batch_size}_q{query_len}"
        model_name = self.speculative_config.model
        if is_warmup:
            COMPILATION_TIME.labels(model_name=model_name, bucket_name=bucket_name).set(
                elapsed
            )
        else:
            NEFF_EXECUTION_COUNT.labels(
                model_name=model_name, bucket_name=bucket_name
            ).inc()
        return draft_token_ids, draft_token_ids
