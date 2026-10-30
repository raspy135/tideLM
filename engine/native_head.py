"""Optional lossless storage and decode projection for the checkpoint's BF16 head.

Keep sign/mantissa bytes and exponent deltas in K-group-major tiles. Unusually
wide exponent groups use an escape table, so every BF16 bit survives packing.
The decode kernel explicitly keeps native MMA kWidth=2: byte-derived ordinary
Triton dot picked kWidth=4 and failed the cancellation regression despite exact
weight reconstruction. See tools/head_packed_gluon.py.
"""
import os

import torch
import torch.nn.functional as F


HEAD_KERNEL_VERSION = 1


def head_kernel():
    value = os.environ.get('DSV41_HEAD_KERNEL', 'off').strip().lower() or 'off'
    if value not in ('off', 'packed'):
        raise ValueError('DSV41_HEAD_KERNEL must be off or packed')
    return value


def validate_config():
    """Fail before allocating a model when a requested head combination is unsupported."""
    mode = head_kernel()
    if mode == 'packed':
        from v41_ref import head_fmt, draft_head_fmt
        if head_fmt() != 'bf16' or os.environ.get('DSV41_HEAD_FP32', '0') == '1':
            raise ValueError('packed head requires native BF16 (HEAD_FMT=bf16, HEAD_FP32=0)')
        from engine.tensor_parallel import tp_draft_head_enabled
        # A separate TP draft shard (DSV41_TP_DRAFT_HEAD=1) is built from the BF16 shard before
        # packing, so a quantized draft format is fine there; without it the drafter would have
        # to quantize the packed verifier head, which has no tensor to quantize.
        allowed = ('off', 'bf16', 'fp8', 'fp4') if tp_draft_head_enabled() else ('off', 'bf16')
        if draft_head_fmt() not in allowed:
            raise ValueError('packed head requires DSV41_DRAFT_HEAD_FMT=off or bf16 '
                             '(fp8/fp4 only with DSV41_TP_DRAFT_HEAD=1)')
    return mode


def make_packed_head(weight):
    # Off does not import experimental Gluon or allocate packing scratch.
    from head_packed_tiles import PackedHead

    class NativeHead(PackedHead):
        def native_logits(self, x):
            shape = x.shape
            if not shape or x.device != self.device or shape[-1] != self.shape[1]:
                raise ValueError('activations must match the packed head device and width')
            xb = x.to(torch.bfloat16).reshape(-1, shape[-1]).contiguous()
            if xb.shape[0] == 0:
                return torch.empty((*shape[:-1], self.shape[0]), dtype=torch.float32,
                                   device=self.device)
            import decode_rows
            if xb.shape[0] <= decode_rows.limit():
                from head_packed_gluon import project
                out = project(xb, self, bn=32, bk=128, warps=4, stages=1)
            else:
                # Bounded BF16 scratch instead of keeping a second full head.
                # Prefill has a different cuBLAS plan from decode in either arm;
                # this path must be qualified separately on actual activations.
                out = torch.empty((xb.shape[0], self.shape[0]), dtype=torch.float32,
                                  device=self.device)
                for first in range(0, self.shape[0], 16384):
                    last = min(first + 16384, self.shape[0])
                    block = self.dequant_rows(first, last)
                    out[:, first:last] = F.linear(xb, block).float()
                    del block
            return out.view(*shape[:-1], self.shape[0])

    return NativeHead(weight)
