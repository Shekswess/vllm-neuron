# SPDX-License-Identifier: Apache-2.0
"""Qwen3-shaped DFlash draft model for Neuron.

The DFlash checkpoint owns only draft weights. The target model provides the
first verified token and selected hidden states; embedding and LM-head weights
omitted by the draft checkpoint are loaded from the target checkpoint.
"""

from __future__ import annotations

import os

import torch
from nkilib.core.utils.common_types import NormType
from torch import nn
from vllm.distributed.parallel_state import get_tp_group

import vllm_neuron.functional as NF
import vllm_neuron.nn as neuron_nn
from vllm_neuron.model.gpt_oss.model_bf16 import GptOssRotaryEmbedding
from vllm_neuron.model.gpt_oss.weight_loaders_bf16 import (
    fused_qkv_bias_loader,
    fused_qkv_weight_loader,
)
from vllm_neuron.model.kv_cache import KVSpec, LayerSpec
from vllm_neuron.model.llama3.eagle3_model import (
    _make_rmsnorm,
    embedding_sharding_padding_weight_loader,
)
from vllm_neuron.model.llama3.model import LlamaMLP
from vllm_neuron.nn.embedding import VocabDimShardedEmbedding
from vllm_neuron.nn.sampler import Sampler
from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint
from vllm_neuron.utils.weight_loader import (
    SafetensorsWeightLoader,
    last_dim_padding_weight_loader,
    scaled_bias_loader,
    set_weight_loader,
    sharding_weight_loader_with_padding,
    with_rank_override,
)

from .config import DFlashConfig


def _feature_projection_loader(
    *,
    target_hidden_size: int,
    draft_hidden_size: int,
    padded_hidden_size: int,
    num_features: int,
) -> SafetensorsWeightLoader:
    """Pad each concatenated target feature independently before projection."""

    def transform(slices: list, rank: int) -> torch.Tensor:
        del rank
        assert len(slices) == 1
        weight = slices[0][:]
        expected = (draft_hidden_size, target_hidden_size * num_features)
        assert tuple(weight.shape) == expected, (
            f"DFlash fc.weight must be {expected}, got {tuple(weight.shape)}"
        )
        features = torch.split(weight, target_hidden_size, dim=1)
        padded_features = [
            torch.nn.functional.pad(f, (0, padded_hidden_size - target_hidden_size))
            for f in features
        ]
        weight = torch.cat(padded_features, dim=1)
        return torch.nn.functional.pad(
            weight, (0, 0, 0, padded_hidden_size - draft_hidden_size)
        )

    return SafetensorsWeightLoader(transform=transform)


class DFlashAttention(nn.Module):
    def __init__(self, config: DFlashConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.dtype = config.torch_dtype
        self.scaling = config.head_dim**-0.5
        self.rms_norm_eps = config.rms_norm_eps
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group

        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_attention_heads_per_rank = self.num_attention_heads // self.world_size
        if self.world_size >= self.num_key_value_heads:
            self.num_key_value_heads_per_rank = 1
            self.num_kv_replicas = self.world_size // self.num_key_value_heads
        else:
            self.num_key_value_heads_per_rank = (
                self.num_key_value_heads // self.world_size
            )
            self.num_kv_replicas = 1
        self.num_key_value_groups = (
            self.num_attention_heads_per_rank // self.num_key_value_heads_per_rank
        )

        self.q_size = self.num_attention_heads_per_rank * self.head_dim
        self.kv_size = self.num_key_value_heads_per_rank * self.head_dim
        qkv_size = self.q_size + 2 * self.kv_size
        self.qkv_split_indices = [self.q_size, self.q_size + self.kv_size]

        self.qkv_proj_weight = nn.Parameter(
            torch.empty(self.hidden_size, qkv_size, dtype=self.dtype)
        )
        self.qkv_proj_bias = nn.Parameter(torch.zeros(qkv_size, dtype=self.dtype))
        self.o_proj_weight = nn.Parameter(
            torch.empty(self.q_size, self.hidden_size, dtype=self.dtype)
        )
        self.o_proj_bias = nn.Parameter(torch.zeros(self.hidden_size, dtype=self.dtype))
        self.q_norm_weight = nn.Parameter(torch.ones(self.head_dim, dtype=self.dtype))
        self.k_norm_weight = nn.Parameter(torch.ones(self.head_dim, dtype=self.dtype))
        self.rotary_emb = GptOssRotaryEmbedding(config)

        self.k_cache = None
        self.v_cache = None
        self._setup_weight_loaders(config)

    def _setup_weight_loaders(self, config: DFlashConfig) -> None:
        set_weight_loader(
            self.qkv_proj_weight,
            fused_qkv_weight_loader(
                q_size=self.q_size,
                kv_size=self.kv_size,
                shard_dim=1,
                num_shards=self.world_size,
                num_kv_heads=self.num_key_value_heads,
                head_dim=self.head_dim,
                hidden_size=self.hidden_size,
                num_kv_replicas=self.num_kv_replicas,
            ),
        )
        set_weight_loader(
            self.qkv_proj_bias,
            fused_qkv_bias_loader(
                q_size=self.q_size,
                kv_size=self.kv_size,
                num_shards=self.world_size,
                num_kv_heads=self.num_key_value_heads,
                head_dim=self.head_dim,
                num_kv_replicas=self.num_kv_replicas,
            ),
        )
        set_weight_loader(
            self.o_proj_weight,
            sharding_weight_loader_with_padding(
                shard_dim=0,
                shard_size=self.q_size,
                num_shards=self.world_size,
                pad_dim=1,
                padded_size=config.hidden_size,
                unpadded_size=config.unpadded_hidden_size,
                is_storage_transposed=True,
            ),
        )
        set_weight_loader(
            self.o_proj_bias,
            with_rank_override(
                scaled_bias_loader(
                    scale=self.world_size, padded_size=config.hidden_size
                ),
                rank=self.rank,
            ),
        )
        set_weight_loader(
            self.q_norm_weight, last_dim_padding_weight_loader(self.head_dim)
        )
        set_weight_loader(
            self.k_norm_weight, last_dim_padding_weight_loader(self.head_dim)
        )

    def project_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        slot_mapping: torch.Tensor,
        block_size: int,
    ) -> None:
        cos, sin = self.rotary_emb(
            context_positions, device=context_states.device, dtype=context_states.dtype
        )
        cos_cache = torch.cat([cos, cos], dim=-1).unsqueeze(0)
        sin_cache = torch.cat([sin, sin], dim=-1).unsqueeze(0)
        _, _, _ = NF.qkv_proj(
            hidden=context_states.unsqueeze(0),
            qkv_weights=self.qkv_proj_weight,
            bias=self.qkv_proj_bias.unsqueeze(0),
            d_head=self.head_dim,
            cos_cache=cos_cache,
            sin_cache=sin_cache,
            num_q_heads=self.num_attention_heads_per_rank,
            num_kv_heads=self.num_key_value_heads_per_rank,
            qk_norm_pre_rope_q_norm=NormType.RMS_NORM,
            qk_norm_pre_rope_k_norm=NormType.RMS_NORM,
            qk_norm_pre_rope_eps=self.rms_norm_eps,
            qk_norm_pre_rope_q_gamma=self.q_norm_weight.unsqueeze(0),
            qk_norm_pre_rope_k_gamma=self.k_norm_weight.unsqueeze(0),
            k_cache=self.k_cache,
            v_cache=self.v_cache,
            use_block_kv=True,
            block_size=block_size,
            slot_mapping=slot_mapping.to(torch.int32),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        attn_metadata: dict,
        active_mask: torch.Tensor,
    ) -> torch.Tensor:
        layer_name = f"layers.{self.layer_idx}.self_attn"
        meta = attn_metadata[layer_name]
        block_table = meta["block_table_tensor"]
        block_size = meta["block_size"]
        batch_size = block_table.shape[0]
        query_len = hidden_states.shape[0] // batch_size
        x = hidden_states.view(batch_size, query_len, self.hidden_size).to(self.dtype)

        cos, sin = self.rotary_emb(positions, device=x.device, dtype=x.dtype)
        half = self.head_dim // 2
        cos_kernel = cos[:, :half].view(batch_size, query_len, half).permute(2, 0, 1)
        sin_kernel = sin[:, :half].view(batch_size, query_len, half).permute(2, 0, 1)

        output = NF.attention_decode(
            X=x,
            W_qkv=self.qkv_proj_weight,
            bias_qkv=self.qkv_proj_bias.unsqueeze(0),
            rmsnorm_QK_pre_rope_enabled=True,
            rmsnorm_QK_pre_rope_eps=self.rms_norm_eps,
            rmsnorm_QK_pre_rope_W_Q=self.q_norm_weight.unsqueeze(0),
            rmsnorm_QK_pre_rope_W_K=self.k_norm_weight.unsqueeze(0),
            cos=cos_kernel,
            sin=sin_kernel,
            rope_contiguous_layout=True,
            active_blocks_table=block_table,
            K_cache=self.k_cache,
            V_cache=self.v_cache,
            attention_mask=active_mask,
            softmax_scale=self.scaling,
            update_cache=False,
            W_out=self.o_proj_weight,
            bias_out=self.o_proj_bias.unsqueeze(0),
        )
        # update_cache=False returns query K/V as well; they intentionally do
        # not enter the persistent context cache because proposal tokens are
        # unverified and DFlash attention is bidirectional within the block.
        output = output[0] if isinstance(output, tuple) else output
        self.tp_group.all_reduce(output)
        return output.reshape(-1, self.hidden_size)


class DFlashDecoderLayer(nn.Module):
    def __init__(self, config: DFlashConfig, layer_idx: int):
        super().__init__()
        self.input_layernorm = _make_rmsnorm(config)
        self.self_attn = DFlashAttention(config, layer_idx)
        self.post_attention_layernorm = _make_rmsnorm(config)
        self.mlp = LlamaMLP(config)
        mlp_shards = self.mlp.mlp_tp_size
        mlp_rank = self.mlp.mlp_tp_rank
        per_rank = self.mlp.intermediate_size_per_rank
        for weight in (self.mlp.gate_proj_weight, self.mlp.up_proj_weight):
            set_weight_loader(
                weight,
                with_rank_override(
                    sharding_weight_loader_with_padding(
                        shard_dim=1,
                        shard_size=per_rank,
                        num_shards=mlp_shards,
                        is_storage_transposed=True,
                        pad_dim=0,
                        padded_size=config.hidden_size,
                        unpadded_size=config.unpadded_hidden_size,
                    ),
                    rank=mlp_rank,
                ),
            )
        set_weight_loader(
            self.mlp.down_proj_weight,
            with_rank_override(
                sharding_weight_loader_with_padding(
                    shard_dim=0,
                    shard_size=per_rank,
                    num_shards=mlp_shards,
                    is_storage_transposed=True,
                    pad_dim=1,
                    padded_size=config.hidden_size,
                    unpadded_size=config.unpadded_hidden_size,
                ),
                rank=mlp_rank,
            ),
        )

    def forward(self, hidden_states, positions, attn_metadata, active_mask):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states, positions, attn_metadata, active_mask
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, is_prefill=False)
        return residual + hidden_states


class DFlashDraftModel(nn.Module):
    def __init__(self, config: DFlashConfig, start_layer_idx: int):
        super().__init__()
        self.config = config
        self.start_layer_idx = start_layer_idx
        self.target_layer_ids = config.target_layer_ids
        if not self.target_layer_ids:
            raise ValueError("DFlash checkpoint must provide target_layer_ids")
        self.num_speculative_tokens = config.block_size - 1
        self.mask_token_id = config.mask_token_id

        self.embed_tokens = VocabDimShardedEmbedding(
            vocab_size=config.vocab_size,
            embed_dim=config.hidden_size,
            dtype=config.torch_dtype,
            tp_group=get_tp_group().device_group,
        )
        set_weight_loader(
            self.embed_tokens.weight,
            embedding_sharding_padding_weight_loader(
                vocab_size_per_rank=self.embed_tokens.vocab_size_per_rank,
                num_shards=self.embed_tokens.tp_size,
                padded_hidden_size=config.hidden_size,
            ),
        )
        self.fc = nn.Linear(
            config.hidden_size * len(self.target_layer_ids),
            config.hidden_size,
            bias=False,
            dtype=config.torch_dtype,
        )
        self.hidden_norm = _make_rmsnorm(config)
        self.layers = nn.ModuleList(
            [
                DFlashDecoderLayer(config, start_layer_idx + i)
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = _make_rmsnorm(config)
        self.lm_head = neuron_nn.ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            dtype=config.torch_dtype,
            gather_output=False,
        )
        set_weight_loader(
            self.lm_head.weight,
            sharding_weight_loader_with_padding(
                shard_dim=0,
                shard_size=self.lm_head.out_features_per_rank,
                num_shards=self.lm_head.tp_size,
                pad_dim=1,
                padded_size=config.hidden_size,
                unpadded_size=config.unpadded_hidden_size,
            ),
        )
        sampling_config = config.neuron_config.on_device_sampling_config
        if sampling_config is None:
            raise ValueError("DFlash on Neuron requires on-device greedy sampling")
        self.sampler = Sampler(
            sampling_config, process_group=get_tp_group().device_group
        )

        set_weight_loader(
            self.fc.weight,
            _feature_projection_loader(
                target_hidden_size=config.target_hidden_size,
                draft_hidden_size=config.unpadded_hidden_size,
                padded_hidden_size=config.hidden_size,
                num_features=len(self.target_layer_ids),
            ),
        )
        set_weight_loader(
            self.hidden_norm.weight, last_dim_padding_weight_loader(config.hidden_size)
        )
        set_weight_loader(
            self.norm.weight, last_dim_padding_weight_loader(config.hidden_size)
        )

    @classmethod
    def from_configs(cls, config, start_layer_idx: int, neuron_config=None):
        return cls(DFlashConfig.from_configs(config, neuron_config), start_layer_idx)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: torch.Tensor,
        query_slot_mapping: torch.Tensor,
        attn_metadata: dict,
        active_mask: torch.Tensor,
        rank: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del rank, query_slot_mapping
        context_states = self.hidden_norm(self.fc(target_hidden_states))
        block_size = next(iter(attn_metadata.values()))["block_size"]
        for layer in self.layers:
            layer.self_attn.project_context_kv(
                context_states,
                context_positions,
                context_slot_mapping,
                block_size,
            )

        hidden_states = self.embed_tokens(input_ids, scatter_tokens=False)
        for layer in self.layers:
            hidden_states = layer(hidden_states, positions, attn_metadata, active_mask)
        hidden_states = self.norm(hidden_states)
        batch_size = next(iter(attn_metadata.values()))["block_table_tensor"].shape[0]
        query_len = hidden_states.shape[0] // batch_size
        proposal_states = hidden_states.view(batch_size, query_len, -1)[:, 1:, :]
        logits = self.lm_head(proposal_states.reshape(-1, self.config.hidden_size))
        return self.sampler(logits).to(torch.int32).view(batch_size, query_len - 1)

    def get_kv_spec(self) -> KVSpec:
        return KVSpec(
            layers=[
                LayerSpec(
                    name=f"layers.{layer.self_attn.layer_idx}.self_attn",
                    num_kv_heads=layer.self_attn.num_key_value_heads_per_rank,
                    head_size=layer.self_attn.head_dim,
                    dtype=layer.self_attn.dtype,
                    sliding_window_size=None,
                    chunk_size=None,
                )
                for layer in self.layers
            ]
        )

    def bind_kv_cache(self, kv_caches) -> None:
        for layer in self.layers:
            name = f"layers.{layer.self_attn.layer_idx}.self_attn"
            layer.self_attn.k_cache, layer.self_attn.v_cache = kv_caches[name]

    def load_weights(self, checkpoint_path: str, device, cache_dir=None) -> None:
        if not os.path.isdir(checkpoint_path):
            from huggingface_hub import snapshot_download

            checkpoint_path = snapshot_download(checkpoint_path, cache_dir=cache_dir)

        mappings = {
            "fc.weight": "fc.weight",
            "hidden_norm.weight": "hidden_norm.weight",
            "norm.weight": "norm.weight",
        }
        for i, layer in enumerate(self.layers):
            prefix = f"layers.{i}"
            model_prefix = f"layers.{i}"
            mappings[f"{model_prefix}.self_attn.qkv_proj_weight"] = [
                f"{prefix}.self_attn.q_proj.weight",
                f"{prefix}.self_attn.k_proj.weight",
                f"{prefix}.self_attn.v_proj.weight",
            ]
            mappings[f"{model_prefix}.self_attn.qkv_proj_bias"] = [
                f"{prefix}.self_attn.q_proj.bias",
                f"{prefix}.self_attn.k_proj.bias",
                f"{prefix}.self_attn.v_proj.bias",
            ]
            mappings[f"{model_prefix}.self_attn.o_proj_weight"] = (
                f"{prefix}.self_attn.o_proj.weight"
            )
            mappings[f"{model_prefix}.self_attn.o_proj_bias"] = (
                f"{prefix}.self_attn.o_proj.bias"
            )
            mappings[f"{model_prefix}.self_attn.q_norm_weight"] = (
                f"{prefix}.self_attn.q_norm.weight"
            )
            mappings[f"{model_prefix}.self_attn.k_norm_weight"] = (
                f"{prefix}.self_attn.k_norm.weight"
            )
            mappings[f"{model_prefix}.input_layernorm.weight"] = (
                f"{prefix}.input_layernorm.weight"
            )
            mappings[f"{model_prefix}.post_attention_layernorm.weight"] = (
                f"{prefix}.post_attention_layernorm.weight"
            )
            mappings[f"{model_prefix}.mlp.gate_proj_weight"] = (
                f"{prefix}.mlp.gate_proj.weight"
            )
            mappings[f"{model_prefix}.mlp.up_proj_weight"] = (
                f"{prefix}.mlp.up_proj.weight"
            )
            mappings[f"{model_prefix}.mlp.down_proj_weight"] = (
                f"{prefix}.mlp.down_proj.weight"
            )

        checkpoint = SafetensorsCheckpoint(checkpoint_path)
        result = checkpoint.load_sharded_pipelined(
            get_tp_group().rank_in_group,
            get_tp_group().world_size,
            self,
            mappings,
            device,
            strict=False,
        )
        missing, unexpected = self.load_state_dict(
            result.state_dict, strict=False, assign=True
        )
        allowed_missing = {"embed_tokens.weight", "lm_head.weight"}
        missing_set = set(missing)
        if missing_set != allowed_missing or unexpected or result.unexpected_keys:
            raise RuntimeError(
                "DFlash draft checkpoint weight mismatch: "
                f"missing={sorted(missing_set)}, unexpected="
                f"{sorted(set(unexpected) | set(result.unexpected_keys))}"
            )

    def load_target_weights(self, checkpoint_path: str, device, cache_dir=None) -> None:
        """Load embedding and LM-head tensors omitted by DFlash checkpoints."""
        if not os.path.isdir(checkpoint_path):
            from huggingface_hub import snapshot_download

            checkpoint_path = snapshot_download(checkpoint_path, cache_dir=cache_dir)
        checkpoint = SafetensorsCheckpoint(checkpoint_path)
        shared_weights = nn.Module()
        shared_weights.add_module("embed_tokens", self.embed_tokens)
        shared_weights.add_module("lm_head", self.lm_head)
        result = checkpoint.load_sharded_pipelined(
            get_tp_group().rank_in_group,
            get_tp_group().world_size,
            shared_weights,
            {
                "embed_tokens.weight": "model.embed_tokens.weight",
                "lm_head.weight": "lm_head.weight",
            },
            device,
            strict=False,
        )
        if result.missing_keys:
            raise RuntimeError(
                "GPT-OSS target is missing DFlash shared weights: "
                f"{result.missing_keys}"
            )
        _, unexpected = self.load_state_dict(
            result.state_dict, strict=False, assign=True
        )
        if unexpected:
            raise RuntimeError(f"Unexpected target weights for DFlash: {unexpected}")
