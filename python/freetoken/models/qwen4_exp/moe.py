from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from freetoken.core import get_global_ctx
from freetoken.distributed import DistributedCommunicator, get_tp_info
from freetoken.kernel.triton.moe_shared_gate import shared_gate_mul_add, shared_gate_sigmoid
from freetoken.layers import LinearRowParallel, silu_and_mul
from freetoken.layers.moe import OffloadMoELayer
from freetoken.models.qwen3_5_moe.moe import Qwen3_5MoE

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


class Qwen4ExpMoE(Qwen3_5MoE):
    """Qwen3_5MoE with the shared-expert gate on triton instead of gemv + sigmoid + mul + add.

    Same weights, same state dict. The gate reduction stays ahead of the routed experts, which may write into ``hidden_states`` in place.

    Shared and routed expert shards produce partial sums, combined before one TP all-reduce.
    """

    def __init__(
        self, config: ModelConfig, layer_id: int | None = None, *, prefix: str = ""
    ) -> None:
        super().__init__(config, layer_id=layer_id, prefix=prefix)
        self._comm = DistributedCommunicator()
        self._tp_size = get_tp_info().size

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        if os.environ.get("FREETOKEN_ROUTER_FP32"):
            from freetoken.kernel.triton.stable_linear import fp32_linear

            router_logits = fp32_linear(hidden_states, self.gate.weight)
        else:
            router_logits = self.gate.forward(hidden_states)
        gate = shared_gate_sigmoid(hidden_states, self.shared_expert_gate.weight.view(-1))
        se, ex = self.shared_expert, self.experts
        if os.environ.get("FREETOKEN_TP_PRECISE") and isinstance(ex, OffloadMoELayer):
            from freetoken.kernel.triton.stable_linear import reduction_linear

            shared = reduction_linear(
                silu_and_mul(se.gate_up_proj.forward(hidden_states)),
                se.down_proj.weight,
            )
            if get_global_ctx().batch.is_prefill:
                routed = ex.prefill_forward(hidden_states, router_logits)
            else:
                routed = ex.decode_forward(hidden_states, router_logits)
            out = shared_gate_mul_add(routed, shared, gate)
            if self._tp_size > 1:
                out = self._comm.all_reduce(out)
            if out.dtype == torch.float64:
                out = out.float()
            if not os.environ.get("FREETOKEN_RESIDUAL_FP32"):
                out = out.to(hidden_states.dtype)
            return out.view(num_tokens, hidden_dim)
        if (
            self._tp_size > 1
            and isinstance(se.down_proj, LinearRowParallel)
            and isinstance(ex, OffloadMoELayer)
        ):
            shared = F.linear(
                silu_and_mul(se.gate_up_proj.forward(hidden_states)), se.down_proj.weight
            )
            if get_global_ctx().batch.is_prefill:
                routed = ex.prefill_forward(hidden_states, router_logits)
            else:
                routed = ex.decode_forward(hidden_states, router_logits)
            out = self._comm.all_reduce(shared_gate_mul_add(routed, shared, gate))
            return out.view(num_tokens, hidden_dim)
        shared = se.forward(hidden_states)
        routed = ex.forward(hidden_states=hidden_states, router_logits=router_logits)
        return shared_gate_mul_add(routed, shared, gate).view(num_tokens, hidden_dim)


__all__ = ["Qwen4ExpMoE"]
