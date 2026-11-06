#!/usr/bin/env python3
"""
Phase 3a: 1D-CNN fatigue classifier (Progress Report I, Sec 4.4).

  input  x   : (B, C, L)   C=4 channels [steer_proxy, lean, steer_rate, pred_error], L=300 (30 s @ 10 Hz)
  input  aux : (B, A)      optional engineered features (Hs, jerk, reversal rate, ...)
  trunk      : 3 x [Conv1d -> BatchNorm -> ReLU -> MaxPool(2)], channels 16/32/64
  pooling    : global average pool over time -> (B, 64)
  head       : concat(aux) -> FC(32) -> ReLU -> Dropout -> FC(1)  -> logit
  output     : logit; P(F_t) = sigmoid(logit)

The model returns LOGITS (not probabilities) because BCEWithLogitsLoss is more
numerically stable than Sigmoid + BCELoss. Mathematically identical to the
"single sigmoid unit" in the report. Apply torch.sigmoid() at inference/export.
~16-20k parameters -> comfortably exportable to ONNX / TFLite.
"""
import torch
import torch.nn as nn


class FatigueCNN(nn.Module):
    def __init__(self, in_ch=4, n_aux=0, widths=(16, 32, 64), kernel=5, head=32, dropout=0.3):
        super().__init__()
        blocks, c = [], in_ch
        for w in widths:
            blocks += [nn.Conv1d(c, w, kernel, padding=kernel // 2),
                       nn.BatchNorm1d(w), nn.ReLU(), nn.MaxPool1d(2)]
            c = w
        self.trunk = nn.Sequential(*blocks)
        self.n_aux = n_aux
        self.head = nn.Sequential(
            nn.Linear(c + n_aux, head), nn.ReLU(), nn.Dropout(dropout), nn.Linear(head, 1))

    def forward(self, x, aux=None):
        h = self.trunk(x).mean(dim=-1)                 # global average pooling
        if self.n_aux:
            h = torch.cat([h, aux], dim=1)
        return self.head(h).squeeze(-1)

    def n_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    for n_aux in (0, 1, 5):
        m = FatigueCNN(n_aux=n_aux)
        out = m(torch.randn(8, 4, 300), torch.randn(8, n_aux) if n_aux else None)
        print(f"n_aux={n_aux}: output {tuple(out.shape)}, params={m.n_params():,}")
