"""
Adapted from https://github.com/mit-han-lab/streaming-llm
"""

from dataclasses import dataclass
from typing import List, Optional

import torch


def slice1d(x, start, end):
    return x[:, start:end, ...]


def slice2d(x, start, end):
    return x[:, :, start:end, ...]


def slice3d(x, start, end):
    return x[:, :, :, start:end, ...]


DIM_TO_SLICE = {
    1: slice1d,
    2: slice2d,
    3: slice3d,
}


@dataclass
class AttentionSinkKVCache:
    attention_sink_size: int = 4
    attention_sink_window_size: int = 1020
    k_seq_dim: int = 2
    v_seq_dim: int = 2
    attention_sink_layer: int = 0
    attention_sink_layer_window: int = 1
    attention_sink_full_attention_layer: List[int] = -1
    attention_sink_head: Optional[int] = None
    attention_sink_heads: Optional[List[int]] = None

    def __post_init__(self):
        self.cache_size = self.attention_sink_size + self.attention_sink_window_size
        self.k_slice = DIM_TO_SLICE[self.k_seq_dim]
        self.v_slice = DIM_TO_SLICE[self.v_seq_dim]

    @staticmethod
    def _infer_head_dim(seq_dim: int, num_dims: int) -> Optional[int]:
        if num_dims < 3:
            return None
        if seq_dim == 1:
            return 2 if num_dims > 2 else None
        return 1

    def __call__(self, past_key_values):
        if past_key_values is None:
            return None
        # tuple_cache = isinstance(past_key_values, tuple)
        # list_cache = isinstance(past_key_values, list)
        # if tuple_cache:
        #     past_key_values = list(past_key_values)
        num_layers = len(past_key_values)
        start = self.attention_sink_layer
        end = start + self.attention_sink_layer_window
        step = 1 if self.attention_sink_layer_window >= 0 else -1
        end = max(0, min(end, num_layers))
        full_layers = self.attention_sink_full_attention_layer
        if isinstance(full_layers, int):
            full_layers = [full_layers]
        elif full_layers is None:
            full_layers = []
        for layer_idx in range(start, end, step):
            if layer_idx in full_layers:
                continue
            k, v = past_key_values[layer_idx]
            seq_len = past_key_values[layer_idx][0].size(self.k_seq_dim)
            if seq_len <= self.cache_size:
                continue
            drop_start = max(0, self.attention_sink_size)
            drop_end = max(drop_start, seq_len - self.attention_sink_window_size)
            if drop_end <= drop_start:
                continue
            heads = None
            if self.attention_sink_heads is not None:
                heads = list(self.attention_sink_heads)
            elif self.attention_sink_head is not None:
                heads = [self.attention_sink_head]
            if heads is not None:
                k_head_dim = self._infer_head_dim(self.k_seq_dim, k.dim())
                v_head_dim = self._infer_head_dim(self.v_seq_dim, v.dim())
                if k_head_dim is None or v_head_dim is None:
                    continue
                num_heads = k.size(k_head_dim)
                norm_heads = []
                for head_idx in heads:
                    if head_idx < 0:
                        head_idx += num_heads
                    if 0 <= head_idx < num_heads:
                        norm_heads.append(head_idx)
                if not norm_heads:
                    continue
                for head_idx in norm_heads:
                    k_idx = [slice(None)] * k.dim()
                    k_idx[self.k_seq_dim] = slice(drop_start, drop_end)
                    k_idx[k_head_dim] = head_idx
                    k[tuple(k_idx)] = 0
                    v_idx = [slice(None)] * v.dim()
                    v_idx[self.v_seq_dim] = slice(drop_start, drop_end)
                    v_idx[v_head_dim] = head_idx
                    v[tuple(v_idx)] = 0
                past_key_values.key_cache[layer_idx] = k
                past_key_values.value_cache[layer_idx] = v

                continue

            new_k = torch.cat(
                        [
                            self.k_slice(k, 0, self.attention_sink_size),
                            self.k_slice(k, seq_len - self.attention_sink_window_size, seq_len),
                        ],
                        dim=self.k_seq_dim,
                    )
            new_v = torch.cat(
                        [
                            self.v_slice(v, 0, self.attention_sink_size),
                            self.v_slice(v, seq_len - self.attention_sink_window_size, seq_len),
                        ],
                        dim=self.v_seq_dim,
                    )
            past_key_values.key_cache[layer_idx] = new_k
            past_key_values.value_cache[layer_idx] = new_v
        # past_key_values._seen_tokens = seq_len
        return past_key_values
    def evict_for_space(self, past_key_values, num_coming):
        if past_key_values is None:
            return None
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        if seq_len + num_coming <= self.cache_size:
            return past_key_values
        return [
            [
                torch.cat(
                    [
                        self.k_slice(k, 0, self.attention_sink_size),
                        self.k_slice(
                            k,
                            seq_len - self.attention_sink_window_size + num_coming,
                            seq_len,
                        ),
                    ],
                    dim=self.k_seq_dim,
                ),
                torch.cat(
                    [
                        self.v_slice(v, 0, self.attention_sink_size),
                        self.v_slice(
                            v,
                            seq_len - self.attention_sink_window_size + num_coming,
                            seq_len,
                        ),
                    ],
                    dim=self.v_seq_dim,
                ),
            ]
            for k, v in past_key_values
        ]

    def evict_range(self, past_key_values, start, end):
        if past_key_values is None:
            return None
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        assert start <= end and end <= seq_len
        return [
            [
                torch.cat(
                    [
                        self.k_slice(k, 0, start),
                        self.k_slice(k, end, seq_len),
                    ],
                    dim=self.k_seq_dim,
                ),
                torch.cat(
                    [
                        self.v_slice(v, 0, start),
                        self.v_slice(v, end, seq_len),
                    ],
                    dim=self.v_seq_dim,
                ),
            ]
            for k, v in past_key_values
        ]
