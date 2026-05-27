"""Stage-3: Sequence-native Transformer with positional encoding.

Tokenises the T-frame window into per-frame state + vision tokens plus K
goal tokens + optional failure-descriptor tokens + learned heatmap query
tokens. A standard ``nn.TransformerEncoder`` lets every token attend to
every other token; heatmap queries pool the input via cross-frame attention
to produce per-patch scalar predictions, which are bilinearly upsampled to
``grid_hw``.

Key difference from UNet ``late_fusion``: late_fusion takes a temporal
*mean* of per-frame ResNet features (each frame independently encoded, then
averaged). This module can learn to weight frames non-uniformly and routes
information across frames with full attention. If a "video" representation
helps beyond what mean-pooling captures, this is the architecture that
should show it.

Modality plumbing matches :class:`BenchmarkMLP`/`BenchmarkConvDec`:
each enabled modality contributes a token (or T tokens for window keys).
Type embeddings tell the encoder what each token is; positional
embeddings tell it where in time/space.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from planner.risk.benchmark_dataset import ModalityConfig
from planner.risk.models.mlp import STATE_DIM, GOAL_DIM, _SmallCNN


# ---------------------------------------------------------------------------
# Positional encodings (sinusoidal, classic)
# ---------------------------------------------------------------------------

def _sinusoidal_pe_1d(n: int, d: int) -> torch.Tensor:
    """1-D sinusoidal PE, shape ``(n, d)``."""
    position = torch.arange(n, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, d, 2, dtype=torch.float32)
                    * -(math.log(10000.0) / d))
    pe = torch.zeros(n, d)
    pe[:, 0::2] = torch.sin(position * div)
    pe[:, 1::2] = torch.cos(position * div)
    return pe


def _sinusoidal_pe_2d(h: int, w: int, d: int) -> torch.Tensor:
    """2-D sinusoidal PE, shape ``(h * w, d)``. ``d`` must be even (uses d/2 for h, d/2 for w)."""
    assert d % 2 == 0, "d_model must be even for 2-D sinusoidal PE"
    dh = d // 2
    dw = d - dh
    pe_h = _sinusoidal_pe_1d(h, dh)              # (h, dh)
    pe_w = _sinusoidal_pe_1d(w, dw)              # (w, dw)
    pe = torch.zeros(h, w, d)
    pe[:, :, :dh] = pe_h.unsqueeze(1).expand(-1, w, -1)
    pe[:, :, dh:] = pe_w.unsqueeze(0).expand(h, -1, -1)
    return pe.reshape(h * w, d)


# ---------------------------------------------------------------------------
# Modality token-types
# ---------------------------------------------------------------------------

# Stable index → meaning. Used by the learned type-embedding so the
# encoder knows what each token represents.
TYPE_IDX = {
    "state":          0,
    "goal":           1,
    "rgb":            2,
    "depth":          3,
    "dino":           4,
    "failure_mode":   5,
    "failure_joints": 6,
    "heatmap_query":  7,
}
N_TYPES = len(TYPE_IDX)


class BenchmarkTransformer(nn.Module):
    """Per-frame tokenisation + self-attention + learned heatmap queries.

    Parameters
    ----------
    modalities : ModalityConfig
        Which inputs to consume.
    grid_hw : (int, int)
        Output heatmap resolution (typically (240, 320)).
    T : int
        Window length. Per-frame keys yield T tokens each.
    K : int
        Goal lookahead count. ``goal`` modality yields K tokens.
    d_model : int
        Token dimensionality.
    n_heads : int
        Attention heads per encoder layer.
    n_layers : int
        Number of encoder layers.
    patch_hw : (int, int)
        Output patch grid; ``Hp * Wp`` heatmap-query tokens are learned and
        the per-token scalar prediction is bilinearly upsampled to grid_hw.
    """

    def __init__(self, *, modalities: ModalityConfig, grid_hw: tuple,
                 T: int = 8, K: int = 3,
                 d_model: int = 256, n_heads: int = 4, n_layers: int = 6,
                 dropout: float = 0.1, patch_hw: tuple = (15, 20),
                 img_embed_dim: int = 128, dino_dim: int = 384):
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError(f"d_model must be even (got {d_model})")
        if not any(getattr(modalities, k) for k in
                   ("state", "goal", "rgb", "depth", "dino",
                    "failure_mode", "failure_joints")):
            raise ValueError("at least one modality must be enabled")

        self.modalities = modalities
        self.grid_hw = tuple(grid_hw)
        self.T = T
        self.K = K
        self.patch_hw = tuple(patch_hw)
        self.d_model = d_model

        # Per-modality input projections.
        if modalities.state:
            self.state_proj = nn.Linear(STATE_DIM, d_model)
        if modalities.goal:
            self.goal_proj = nn.Linear(GOAL_DIM, d_model)
        if modalities.rgb:
            # Small per-frame CNN → d_model token. (Use DINOv2 instead via
            # modalities.dino for stronger features.)
            self.rgb_encoder = _SmallCNN(3, d_model)
        if modalities.depth:
            self.depth_encoder = _SmallCNN(1, d_model)
        if modalities.dino:
            self.dino_proj = nn.Linear(dino_dim, d_model)
        if modalities.failure_mode:
            self.fmode_proj = nn.Linear(5, d_model)
        if modalities.failure_joints:
            self.fjoints_proj = nn.Linear(7, d_model)

        # Type embeddings: one learned vector per token-type.
        self.type_emb = nn.Embedding(N_TYPES, d_model)

        # Positional encodings (buffers — fixed sinusoidal).
        self.register_buffer("temporal_pe",
                             _sinusoidal_pe_1d(T, d_model), persistent=False)
        self.register_buffer("goal_pe",
                             _sinusoidal_pe_1d(K, d_model), persistent=False)
        Hp, Wp = self.patch_hw
        self.register_buffer("spatial_pe",
                             _sinusoidal_pe_2d(Hp, Wp, d_model), persistent=False)

        # Learned heatmap query tokens — one per patch in the patch grid.
        self.heatmap_queries = nn.Parameter(
            torch.randn(Hp * Wp, d_model) * 0.02)

        # Transformer encoder. Pre-LN, GELU, batch_first.
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

        # Output head: scalar per query.
        self.out_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
        )

    # ----------------------------------------------------------------- utils

    def _add_type_and_pe(self, t: torch.Tensor, type_name: str,
                         pe: torch.Tensor | None) -> torch.Tensor:
        """Add the type embedding (and optional positional embedding) to ``t``.

        ``t`` shape: ``(B, N, d)`` where N is the number of tokens of this
        type. ``pe`` shape: ``(N, d)`` or ``None`` for single tokens.
        """
        type_vec = self.type_emb(torch.tensor(TYPE_IDX[type_name],
                                              device=t.device))
        out = t + type_vec.view(1, 1, -1)
        if pe is not None:
            out = out + pe[None, :t.shape[1], :]
        return out

    def _build_tokens(self, batch: dict):
        """Build all input tokens + heatmap-query tokens.

        Returns
        -------
        kv_tokens : (B, N_kv, d)
        queries   : (B, N_q,  d)
        """
        ref = None
        for k in ("state_window", "rgb_window", "depth_window",
                  "dino_window", "goal", "failure_mode", "failure_joints"):
            if k in batch and torch.is_tensor(batch[k]):
                ref = batch[k]
                break
        if ref is None:
            raise RuntimeError("no recognised input key in batch")
        B = ref.shape[0]

        parts: list = []
        if self.modalities.state:
            sw = batch["state_window"]                            # (B, T, 18)
            tok = self.state_proj(sw)                             # (B, T, d)
            parts.append(self._add_type_and_pe(tok, "state", self.temporal_pe))
        if self.modalities.goal:
            gl = batch["goal"]                                    # (B, K, 11)
            tok = self.goal_proj(gl)
            parts.append(self._add_type_and_pe(tok, "goal", self.goal_pe))
        if self.modalities.rgb:
            rgb = batch["rgb_window"]                             # (B, T, 3, H, W)
            T = rgb.shape[1]
            flat = rgb.reshape(B * T, 3, rgb.shape[3], rgb.shape[4])
            emb = self.rgb_encoder(flat).view(B, T, self.d_model)
            parts.append(self._add_type_and_pe(emb, "rgb", self.temporal_pe))
        if self.modalities.depth:
            dp = batch["depth_window"]                            # (B, T, 1, H, W)
            T = dp.shape[1]
            flat = dp.reshape(B * T, 1, dp.shape[3], dp.shape[4])
            emb = self.depth_encoder(flat).view(B, T, self.d_model)
            parts.append(self._add_type_and_pe(emb, "depth", self.temporal_pe))
        if self.modalities.dino:
            dn = batch["dino_window"]                             # (B, T, 384)
            tok = self.dino_proj(dn)
            parts.append(self._add_type_and_pe(tok, "dino", self.temporal_pe))
        if self.modalities.failure_mode:
            fm = batch["failure_mode"].unsqueeze(1)               # (B, 1, 5)
            tok = self.fmode_proj(fm)
            parts.append(self._add_type_and_pe(tok, "failure_mode", None))
        if self.modalities.failure_joints:
            fj = batch["failure_joints"].unsqueeze(1)             # (B, 1, 7)
            tok = self.fjoints_proj(fj)
            parts.append(self._add_type_and_pe(tok, "failure_joints", None))

        kv = torch.cat(parts, dim=1)                              # (B, N_kv, d)

        Q = self.patch_hw[0] * self.patch_hw[1]
        q = self.heatmap_queries.unsqueeze(0).expand(B, -1, -1)
        q = self._add_type_and_pe(q, "heatmap_query", self.spatial_pe)

        return kv, q

    # ---------------------------------------------------------------- forward

    def forward(self, batch: dict) -> dict:
        kv, q = self._build_tokens(batch)
        # Concat input tokens + heatmap queries, run one self-attention stack.
        # The queries learn to pool the kv content via attention.
        N_kv = kv.shape[1]
        all_tokens = torch.cat([kv, q], dim=1)                    # (B, N_kv+N_q, d)
        encoded = self.encoder(all_tokens)
        q_out = encoded[:, N_kv:, :]                              # (B, N_q, d)
        scalars = self.out_head(q_out).squeeze(-1)                # (B, N_q)

        Hp, Wp = self.patch_hw
        patches = scalars.view(-1, 1, Hp, Wp)                     # (B, 1, Hp, Wp)
        pred = torch.nn.functional.interpolate(
            patches, size=self.grid_hw, mode="bilinear", align_corners=False)
        return {"pred": pred.squeeze(1)}
