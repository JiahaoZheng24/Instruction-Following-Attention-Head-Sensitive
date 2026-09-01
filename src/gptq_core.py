"""Masked GPTQ core: standard GPTQ (act-order, grouped, sym) with an optional
per-weight protection mask held out INSIDE the column loop.

Protected entries keep their fp value and propagate ZERO error — the
compensation never "repairs around" them and never assumes they were
quantized. This is the same surgery TaCQ/SPQR perform (in-loop exemption),
as opposed to v1's post-hoc restoration (protect_eval.py), which conflicts
with compensation already distributed across columns.

Faithful port of the IST-DASLab GPTQ fasterquant loop (Apache-2.0), with:
  - mask: bool [rows, cols], True = keep fp16 (exempt from quantization)
  - fake-quant output: returns the dequantized fp weight matrix (values lie
    exactly on the b-bit grid except protected entries) — saved as a normal
    HF checkpoint; packing is mechanical and orthogonal to the science.

Known deviation from gptqmodel's pipeline (documented in the paper):
module Hessians within a layer are captured in ONE pre-quantization forward
(gptqmodel re-captures o_proj/down_proj after quantizing earlier modules).
All v2 arms therefore compare against the v2 no-protection arm, not against
the W0 gptqmodel checkpoints.
"""
import torch


class MaskedGPTQ:
    def __init__(self, layer: torch.nn.Linear):
        self.layer = layer
        W = layer.weight.data
        self.rows, self.columns = W.shape
        self.dev = W.device
        self.H = torch.zeros((self.columns, self.columns), device=self.dev,
                             dtype=torch.float32)
        self.nsamples = 0

    @torch.no_grad()
    def add_batch(self, inp: torch.Tensor):
        """inp: [..., columns] input activations of this linear."""
        x = inp.reshape(-1, self.columns).t().float()  # [cols, N]
        n = x.shape[1]
        self.H *= self.nsamples / (self.nsamples + n)
        self.nsamples += n
        x = x * (2.0 / self.nsamples) ** 0.5
        self.H += x @ x.t()

    @torch.no_grad()
    def quantize(self, bits=3, group_size=128, sym=True, actorder=True,
                 percdamp=0.05, blocksize=128, mask: torch.Tensor | None = None):
        W = self.layer.weight.data.clone().float()
        H = self.H.clone()
        M = mask.to(self.dev) if mask is not None else None

        dead = torch.diag(H) == 0
        H[dead, dead] = 1.0
        W[:, dead] = 0

        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            if M is not None:
                M = M[:, perm]

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        Hinv = torch.linalg.cholesky(H, upper=True)

        maxq = 2 ** bits - 1
        zero = (maxq + 1) / 2  # sym
        assert sym, "protocol is sym=True"
        Q = torch.zeros_like(W)
        scale = None

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1
            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                col = i1 + i
                w = W1[:, i]
                d = Hinv1[i, i]
                if group_size != -1 and col % group_size == 0:
                    g = W[:, col:col + group_size]
                    xmax = g.abs().max(dim=1).values.clamp(min=1e-8)
                    scale = (2.0 * xmax / maxq).unsqueeze(1)  # [rows,1]
                q = torch.clamp(torch.round(w.unsqueeze(1) / scale) + zero,
                                0, maxq)
                q = (scale * (q - zero)).flatten()
                if M is not None:
                    m = M[:, col]
                    q[m] = w[m]              # exempt: keep fp, zero error
                Q1[:, i] = q
                err = (w - q) / d
                W1[:, i:] -= err.unsqueeze(1) * Hinv1[i, i:].unsqueeze(0)
                Err1[:, i] = err

            Q[:, i1:i2] = Q1
            W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]

        if actorder:
            invperm = torch.argsort(perm)
            Q = Q[:, invperm]

        self.layer.weight.data = Q.to(self.layer.weight.dtype)
        del H, Hinv
        return Q

    def free(self):
        self.H = None
        torch.cuda.empty_cache()
