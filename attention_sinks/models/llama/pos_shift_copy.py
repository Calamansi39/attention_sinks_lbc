"""
Adapted from https://github.com/mit-han-lab/streaming-llm
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from transformers.models.llama.modeling_llama import repeat_kv, rotate_half

__all__ = ["llama_pos_shift_attention_forward"]


def apply_rotary_pos_emb_single(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.LongTensor] = None,
) -> torch.Tensor:
    # Normalize cos/sin to [bs, seq_len, head_dim] across transformer versions.
    if cos.dim() == 4:
        cos = cos.squeeze(1)
    if sin.dim() == 4:
        sin = sin.squeeze(1)
    if cos.dim() == 2:
        cos = cos.unsqueeze(0)
    if sin.dim() == 2:
        sin = sin.unsqueeze(0)

    if cos.size(0) == 1 and x.size(0) > 1:
        cos = cos.expand(x.size(0), -1, -1)
        sin = sin.expand(x.size(0), -1, -1)

    if cos.size(1) != x.size(2):
        if position_ids is None:
            raise RuntimeError(
                f"RoPE length mismatch: cos_seq={cos.size(1)} vs x_seq={x.size(2)} and no position_ids provided."
            )
        if position_ids.dim() == 1:
            position_ids = position_ids.unsqueeze(0)
        if position_ids.size(0) == 1 and x.size(0) > 1:
            position_ids = position_ids.expand(x.size(0), -1)
        gather_index = position_ids.unsqueeze(-1).expand(-1, -1, cos.size(-1))
        cos = torch.gather(cos, dim=1, index=gather_index)
        sin = torch.gather(sin, dim=1, index=gather_index)

    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (x * cos) + (rotate_half(x) * sin)


def _compute_rope_cos_sin(
    self,
    value_states: torch.Tensor,
    position_ids: torch.LongTensor,
    fallback_position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if hasattr(self, "rotary_emb"):
        try:
            return self.rotary_emb(value_states, position_ids)
        except TypeError:
            # Backward-compatible path for older transformers signatures.
            seq_len = int(position_ids.max().item()) + 1
            return self.rotary_emb(value_states, seq_len=seq_len)

    if fallback_position_embeddings is None:
        raise AttributeError("`position_embeddings` is required when `rotary_emb` is unavailable.")
    return fallback_position_embeddings

def _normalize_head_indices(heads, num_heads: int) -> Optional[List[int]]:
    if heads is None:
        return None
    if isinstance(heads, int):
        heads = [heads]
    normalized: List[int] = []
    for head_idx in heads:
        head_idx = int(head_idx)
        if head_idx < 0:
            head_idx += num_heads
        if 0 <= head_idx < num_heads and head_idx not in normalized:
            normalized.append(head_idx)
    return normalized


def _layer_in_streaming_window(config, layer_idx: Optional[int]) -> bool:
    if layer_idx is None:
        return True

    start = int(getattr(config, "attention_sink_layer", 0))
    window = int(getattr(config, "attention_sink_layer_window", 1))
    num_layers = int(getattr(config, "num_hidden_layers", layer_idx + 1))

    full_layers = getattr(config, "attention_sink_full_attention_layer", -1)
    if isinstance(full_layers, int):
        full_layers = [full_layers]
    elif full_layers is None:
        full_layers = []

    if layer_idx in full_layers:
        return False
    if window == 0:
        return False

    end = start + window
    step = 1 if window >= 0 else -1
    end = max(0, min(end, num_layers))
    return layer_idx in range(start, end, step)


def _resolve_kv_head_partitions(self, device) -> Tuple[torch.LongTensor, torch.LongTensor]:
    num_kv_heads = getattr(self, "num_key_value_heads", None)
    if num_kv_heads is None:
        num_kv_heads = int(getattr(self.config, "num_key_value_heads"))
    layer_idx = getattr(self, "layer_idx", None)
    apply_streaming = _layer_in_streaming_window(self.config, layer_idx)

    all_kv_heads = list(range(num_kv_heads))
    if not apply_streaming:
        full_kv_heads = all_kv_heads
        stream_kv_heads = []
    else:
        full_from_config = _normalize_head_indices(
            getattr(self.config, "attention_sink_full_attention_heads", None),
            num_kv_heads,
        )
        if full_from_config is not None:
            full_kv_heads = full_from_config
            full_kv_set = set(full_kv_heads)
            stream_kv_heads = [idx for idx in all_kv_heads if idx not in full_kv_set]
        else:
            stream_heads = _normalize_head_indices(
                getattr(self.config, "attention_sink_heads", None),
                num_kv_heads,
            )
            if stream_heads is None:
                stream_heads = _normalize_head_indices(
                    getattr(self.config, "attention_sink_head", None),
                    num_kv_heads,
                )
            if stream_heads is None:
                stream_heads = all_kv_heads
            stream_kv_heads = stream_heads
            stream_kv_set = set(stream_kv_heads)
            full_kv_heads = [idx for idx in all_kv_heads if idx not in stream_kv_set]

    full_kv_indices = torch.tensor(full_kv_heads, dtype=torch.long, device=device)
    stream_kv_indices = torch.tensor(stream_kv_heads, dtype=torch.long, device=device)
    return full_kv_indices, stream_kv_indices


def _resolve_query_indices(kv_indices: torch.LongTensor, num_key_value_groups: int) -> torch.LongTensor:
    if kv_indices.numel() == 0:
        return kv_indices.new_empty((0,), dtype=torch.long)
    local_query = torch.arange(num_key_value_groups, device=kv_indices.device)
    return (kv_indices.unsqueeze(1) * num_key_value_groups + local_query.unsqueeze(0)).reshape(-1)


def _select_heads(hidden_states: torch.Tensor, head_indices: torch.LongTensor) -> torch.Tensor:
    if head_indices.numel() == 0:
        return hidden_states[:, :0, :, :]
    return hidden_states.index_select(1, head_indices)


def _empty_kv_states(
    bsz: int,
    num_heads: int,
    head_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.empty(bsz, num_heads, 0, head_dim, device=device, dtype=dtype)


def _unpack_duo_cache(
    self,
    past_key_value,
    bsz: int,
    full_kv_indices: torch.LongTensor,
    stream_kv_indices: torch.LongTensor,
    dtype: torch.dtype,
):
    num_full_kv_head = full_kv_indices.numel()
    num_stream_kv_head = stream_kv_indices.numel()
    head_dim = self.head_dim
    device = full_kv_indices.device

    empty_full = _empty_kv_states(bsz, num_full_kv_head, head_dim, device, dtype)
    empty_stream = _empty_kv_states(bsz, num_stream_kv_head, head_dim, device, dtype)

    if past_key_value is None:
        return empty_full, empty_full, empty_stream, empty_stream

    if (
        isinstance(past_key_value, tuple)
        and len(past_key_value) == 2
        and all(torch.is_tensor(item) for item in past_key_value)
        and past_key_value[0].dim() == 4
        and past_key_value[1].dim() == 4
        and past_key_value[0].size(0) == 2 * bsz
        and past_key_value[1].size(0) == 2 * bsz
    ):
        first, second = past_key_value
        first_is_stream = first.size(1) == num_stream_kv_head and second.size(1) == num_full_kv_head
        first_is_full = first.size(1) == num_full_kv_head and second.size(1) == num_stream_kv_head

        if first_is_stream:
            stream_pack, full_pack = first, second
            self._duo_stream_first = True
        elif first_is_full:
            full_pack, stream_pack = first, second
            self._duo_stream_first = False
        else:
            stream_first = bool(getattr(self, "_duo_stream_first", num_stream_kv_head > 0))
            if stream_first:
                stream_pack, full_pack = first, second
            else:
                full_pack, stream_pack = first, second
            self._duo_stream_first = stream_first

        full_key_states = full_pack[:bsz]
        full_value_states = full_pack[bsz:]
        stream_key_states = stream_pack[:bsz]
        stream_value_states = stream_pack[bsz:]
        return full_key_states, full_value_states, stream_key_states, stream_value_states

    if (
        isinstance(past_key_value, tuple)
        and len(past_key_value) == 2
        and all(torch.is_tensor(item) for item in past_key_value)
    ):
        past_key_states, past_value_states = past_key_value
        full_key_states = _select_heads(past_key_states, full_kv_indices)
        full_value_states = _select_heads(past_value_states, full_kv_indices)
        stream_key_states = _select_heads(past_key_states, stream_kv_indices)
        stream_value_states = _select_heads(past_value_states, stream_kv_indices)
        return full_key_states, full_value_states, stream_key_states, stream_value_states

    return empty_full, empty_full, empty_stream, empty_stream


def _pack_duo_cache(self, full_key_states, full_value_states, stream_key_states, stream_value_states):
    full_pack = torch.cat([full_key_states, full_value_states], dim=0)
    stream_pack = torch.cat([stream_key_states, stream_value_states], dim=0)
    if stream_key_states.size(1) > 0:
        self._duo_stream_first = True
        return stream_pack, full_pack
    self._duo_stream_first = False
    return full_pack, stream_pack


def _compress_streaming_cache(
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    sink_size: int,
    window_size: int,
):
    if key_states.size(1) == 0:
        return key_states, value_states

    sink_size = max(0, int(sink_size))
    window_size = max(0, int(window_size))
    cache_size = sink_size + window_size
    if cache_size <= 0:
        return key_states[:, :, :0, :], value_states[:, :, :0, :]
    if key_states.size(2) <= cache_size:
        return key_states, value_states

    key_parts = []
    value_parts = []
    if sink_size > 0:
        key_parts.append(key_states[:, :, :sink_size, :])
        value_parts.append(value_states[:, :, :sink_size, :])
    if window_size > 0:
        key_parts.append(key_states[:, :, -window_size:, :])
        value_parts.append(value_states[:, :, -window_size:, :])

    return torch.cat(key_parts, dim=2), torch.cat(value_parts, dim=2)


def _group_attention(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    num_key_value_groups: int,
    head_dim: int,
    attention_mask: Optional[torch.Tensor],
    output_attentions: bool,
    attention_dropout: float = 0.0,
    training: bool = False,
):
    if query_states.size(1) == 0:
        return None, None

    key_states = repeat_kv(key_states, num_key_value_groups)
    value_states = repeat_kv(value_states, num_key_value_groups)

    q_len = query_states.size(-2)
    kv_len = key_states.size(-2)
    causal_mask = None
    if attention_mask is not None and attention_mask.dim() == 4 and attention_mask.size(-1) >= kv_len:
        causal_mask = attention_mask[..., -kv_len:]

    # Match HF LlamaSdpaAttention behavior: use SDPA kernel when attention weights are not requested.
    if not output_attentions:
        # Avoid known non-contiguous + custom mask issue on some torch/cuda combinations.
        if query_states.device.type == "cuda" and causal_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        is_causal = True if causal_mask is None and q_len > 1 else False
        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=attention_dropout if training else 0.0,
            is_causal=is_causal,
        )
        return attn_output, None

    # Fallback for output_attentions=True, because SDPA does not return attention weights.
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)

    past_len = kv_len - q_len
    query_pos = torch.arange(q_len, device=attn_weights.device).view(1, 1, q_len, 1)
    key_pos = torch.arange(kv_len, device=attn_weights.device).view(1, 1, 1, kv_len)
    causal = key_pos > (past_len + query_pos)
    attn_weights = attn_weights.masked_fill(causal, torch.finfo(attn_weights.dtype).min)

    if causal_mask is not None:
        attn_weights = attn_weights + causal_mask

    attn_probs = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_output = torch.matmul(attn_probs, value_states)
    return attn_output, attn_probs


def llama_pos_shift_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    padding_mask: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()
    device = hidden_states.device

    if cache_position is None:
        cache_position = kwargs.pop("cache_position", None)
    if position_embeddings is None:
        position_embeddings = kwargs.pop("position_embeddings", None)

    num_heads = getattr(self, "num_heads", None)
    if num_heads is None:
        num_heads = int(getattr(self.config, "num_attention_heads"))
    num_key_value_heads = getattr(self, "num_key_value_heads", None)
    if num_key_value_heads is None:
        num_key_value_heads = int(getattr(self.config, "num_key_value_heads"))
    num_key_value_groups = getattr(self, "num_key_value_groups", None)
    if num_key_value_groups is None:
        num_key_value_groups = num_heads // num_key_value_heads
    hidden_size = getattr(self, "hidden_size", None)
    if hidden_size is None:
        hidden_size = int(getattr(self.config, "hidden_size"))

    if self.config.pretraining_tp > 1:
        key_value_slicing = (num_key_value_heads * self.head_dim) // self.config.pretraining_tp
        query_slices = self.q_proj.weight.split(
            (num_heads * self.head_dim) // self.config.pretraining_tp,
            dim=0,
        )
        key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
        value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

        query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
        query_states = torch.cat(query_states, dim=-1)

        key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
        key_states = torch.cat(key_states, dim=-1)

        value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
        value_states = torch.cat(value_states, dim=-1)
          
    else:
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

    # 变形为 [bs, heads, seq, head_dim]
    query_states = query_states.view(bsz, q_len, num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, num_key_value_heads, self.head_dim).transpose(1, 2)

    full_kv_indices, stream_kv_indices = _resolve_kv_head_partitions(self, device)
    full_q_indices = _resolve_query_indices(full_kv_indices, num_key_value_groups)
    stream_q_indices = _resolve_query_indices(stream_kv_indices, num_key_value_groups)

    self.num_full_attn_head = int(full_kv_indices.numel())
    self.num_streaming_attn_head = int(stream_kv_indices.numel())
    self.num_full_query_head = int(full_q_indices.numel())
    self.num_streaming_query_head = int(stream_q_indices.numel())

    cache_obj = None
    if past_key_value is not None and hasattr(past_key_value, "update") and hasattr(past_key_value, "key_cache"):
        cache_obj = past_key_value
        past_key_value = getattr(self, "_duo_past_key_value", None)

    (
        past_full_key_states,
        past_full_value_states,
        past_stream_key_states,
        past_stream_value_states,
    ) = _unpack_duo_cache(
        self,
        past_key_value,
        bsz,
        full_kv_indices,
        stream_kv_indices,
        dtype=key_states.dtype,
    )

    past_seq_len = max(past_full_key_states.size(2), past_stream_key_states.size(2))
    # With transformers>=4.45 and legacy cache conversion, incoming position_ids can be inconsistent
    # with our custom duo cache. Rebuild from actual cached length for decoding.
    if cache_obj is not None:
        position_ids = torch.arange(
            past_seq_len,
            past_seq_len + q_len,
            dtype=torch.long,
            device=device,
        ).unsqueeze(0)
    elif position_ids is None:
        if cache_position is not None:
            position_ids = cache_position.unsqueeze(0) if cache_position.dim() == 1 else cache_position
        else:
            position_ids = torch.arange(
                past_seq_len,
                past_seq_len + q_len,
                dtype=torch.long,
                device=device,
            ).unsqueeze(0)
    elif position_ids.dim() == 1:
        position_ids = position_ids.unsqueeze(0)
    if position_ids.size(0) == 1 and bsz > 1:
        position_ids = position_ids.expand(bsz, -1)

    full_key_states = torch.cat([past_full_key_states, _select_heads(key_states, full_kv_indices)], dim=2)
    full_value_states = torch.cat([past_full_value_states, _select_heads(value_states, full_kv_indices)], dim=2)
    stream_key_states = torch.cat([past_stream_key_states, _select_heads(key_states, stream_kv_indices)], dim=2)
    stream_value_states = torch.cat([past_stream_value_states, _select_heads(value_states, stream_kv_indices)], dim=2)

    query_cos, query_sin = _compute_rope_cos_sin(
        self,
        value_states,
        position_ids,
        fallback_position_embeddings=position_embeddings,
    )
    query_states = apply_rotary_pos_emb_single(query_states, query_cos, query_sin, position_ids)

    if full_key_states.size(2) > 0:
        full_position_ids = torch.arange(full_key_states.size(2), dtype=torch.long, device=device).unsqueeze(0)
        if full_position_ids.size(0) == 1 and bsz > 1:
            full_position_ids = full_position_ids.expand(bsz, -1)
        full_cos, full_sin = _compute_rope_cos_sin(self, value_states, full_position_ids)
        full_key_states = apply_rotary_pos_emb_single(full_key_states, full_cos, full_sin, full_position_ids)
    if stream_key_states.size(2) > 0:
        stream_position_ids = torch.arange(stream_key_states.size(2), dtype=torch.long, device=device).unsqueeze(0)
        if stream_position_ids.size(0) == 1 and bsz > 1:
            stream_position_ids = stream_position_ids.expand(bsz, -1)
        stream_cos, stream_sin = _compute_rope_cos_sin(self, value_states, stream_position_ids)
        stream_key_states = apply_rotary_pos_emb_single(stream_key_states, stream_cos, stream_sin, stream_position_ids)

    attn_output = torch.zeros_like(query_states)
    attn_weights = None

    if full_q_indices.numel() > 0:
        full_query_states = query_states.index_select(1, full_q_indices)
        full_attn_output, _ = _group_attention(
            full_query_states,
            full_key_states,
            full_value_states,
            num_key_value_groups,
            self.head_dim,
            attention_mask,
            output_attentions,
            attention_dropout=getattr(self, "attention_dropout", 0.0),
            training=self.training,
        )
        attn_output.index_copy_(1, full_q_indices, full_attn_output)

    if stream_q_indices.numel() > 0:
        stream_query_states = query_states.index_select(1, stream_q_indices)
        stream_attn_output, _ = _group_attention(
            stream_query_states,
            stream_key_states,
            stream_value_states,
            num_key_value_groups,
            self.head_dim,
            attention_mask,
            output_attentions,
            attention_dropout=getattr(self, "attention_dropout", 0.0),
            training=self.training,
        )
        attn_output.index_copy_(1, stream_q_indices, stream_attn_output)

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, hidden_size)

    if self.config.pretraining_tp > 1:
        attn_output = attn_output.split(hidden_size // self.config.pretraining_tp, dim=2)
        o_proj_slices = self.o_proj.weight.split(hidden_size // self.config.pretraining_tp, dim=1)
        attn_output = sum(F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp))
    else:
        attn_output = self.o_proj(attn_output)

    if _layer_in_streaming_window(self.config, getattr(self, "layer_idx", None)):
        sink_size = getattr(self.config, "attention_sink_size", 4)
        window_size = getattr(self.config, "attention_sink_window_size", 1020)
        stream_key_states, stream_value_states = _compress_streaming_cache(
            stream_key_states,
            stream_value_states,
            sink_size=sink_size,
            window_size=window_size,
        )

    past_key_value = (
        _pack_duo_cache(
            self,
            full_key_states,
            full_value_states,
            stream_key_states,
            stream_value_states,
        )
        if use_cache
        else None
    )

    if not output_attentions:
        attn_weights = None

    if cache_obj is not None:
        if use_cache:
            self._duo_past_key_value = past_key_value
        # transformers>=4.45 decoder layers always unpack 3 values from attention
        # (hidden_states, attn_weights, present_key_value).
        return attn_output, attn_weights, cache_obj

    return attn_output, attn_weights, past_key_value
