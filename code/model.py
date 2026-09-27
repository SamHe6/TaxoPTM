import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class SiteRelativeEncoder(nn.Module):
    def __init__(self, units: int, num_buckets: int = 16, max_distance: int = 32):
        super().__init__()
        self.units = units
        self.num_buckets = num_buckets
        self.max_distance = max_distance

        self.proj = nn.Sequential(
            nn.LazyLinear(units),
            nn.GELU(),
            nn.LayerNorm(units),
        )
        self.scalar_mlp = nn.Sequential(
            nn.Linear(2, units),
            nn.GELU(),
            nn.Linear(units, units),
        )
        self.bucket_emb = nn.Embedding(num_buckets, units)
        self.side_emb = nn.Embedding(3, units)
        self.out_norm = nn.LayerNorm(units)

    def _distance_buckets(self, rel_pos: torch.Tensor) -> torch.Tensor:
        abs_pos = rel_pos.abs().clamp(max=self.max_distance)
        bucket = torch.div(abs_pos * self.num_buckets, self.max_distance + 1, rounding_mode="floor")
        return bucket.clamp(max=self.num_buckets - 1).long()

    def forward(self, x_emb: torch.Tensor, x_mask: torch.Tensor):
        valid_mask = x_mask.bool()
        lengths = x_mask.sum(dim=1).long().clamp_min(1)
        centers = (lengths // 2).long()

        bsz, seq_len, _ = x_emb.shape
        idx = torch.arange(seq_len, device=x_emb.device).view(1, seq_len).expand(bsz, seq_len)
        rel_pos = idx - centers.view(-1, 1)

        signed = rel_pos.float() / lengths.view(-1, 1).float()
        absolute = rel_pos.abs().float() / lengths.view(-1, 1).float()
        scalar_feat = torch.stack([signed, absolute], dim=-1)

        side = torch.where(
            rel_pos < 0,
            torch.zeros_like(rel_pos),
            torch.where(rel_pos == 0, torch.ones_like(rel_pos), torch.full_like(rel_pos, 2)),
        )
        bucket = self._distance_buckets(rel_pos)

        h = self.proj(x_emb)
        h = h + self.scalar_mlp(scalar_feat)
        h = h + self.bucket_emb(bucket)
        h = h + self.side_emb(side)
        h = self.out_norm(h)
        h = h * valid_mask.unsqueeze(-1)

        return {
            "h": h,
            "valid_mask": valid_mask,
            "centers": centers,
            "rel_pos": rel_pos,
            "bucket": bucket,
        }


class CenterConditionedAttention(nn.Module):
    def __init__(self, units: int, heads: int = 4, num_buckets: int = 16, dropout: float = 0.1):
        super().__init__()
        assert units % heads == 0, "units must be divisible by heads"
        self.units = units
        self.heads = heads
        self.head_dim = units // heads
        self.scale = math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(units, units)
        self.k_proj = nn.Linear(units, units)
        self.v_proj = nn.Linear(units, units)
        self.out_proj = nn.Linear(units, units)
        self.dist_bias = nn.Embedding(num_buckets, heads)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(units)

    def forward(self, h: torch.Tensor, valid_mask: torch.Tensor, centers: torch.Tensor, bucket: torch.Tensor):
        bsz, seq_len, _ = h.shape
        center_h = h.gather(1, centers.view(-1, 1, 1).expand(-1, 1, h.size(-1)))

        q = self.q_proj(center_h).view(bsz, 1, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(bsz, seq_len, self.heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(bsz, seq_len, self.heads, self.head_dim).transpose(1, 2)

        attn_logits = (q * k).sum(dim=-1) / self.scale
        attn_logits = attn_logits + self.dist_bias(bucket).permute(0, 2, 1)
        attn_logits = attn_logits.masked_fill(~valid_mask.unsqueeze(1), float("-inf"))

        attn = torch.softmax(attn_logits, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0, posinf=0.0, neginf=0.0)
        attn = self.dropout(attn)

        context = torch.matmul(attn.unsqueeze(2), v).squeeze(2)
        context = context.transpose(1, 2).reshape(bsz, self.units)
        context = self.norm(self.out_proj(context) + center_h.squeeze(1))

        return {
            "context_vec": context,
            "attn_weights": attn,
            "center_repr": center_h.squeeze(1),
        }


class MotifPrototypeBank(nn.Module):
    def __init__(self, units: int, num_prototypes: int = 12, temperature: float = 0.25):
        super().__init__()
        self.temperature = temperature
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, units))
        nn.init.xavier_uniform_(self.prototypes)

    def forward(self, context_vec: torch.Tensor):
        context_n = F.normalize(context_vec, dim=-1)
        proto_n = F.normalize(self.prototypes, dim=-1)
        logits = torch.matmul(context_n, proto_n.t()) / self.temperature
        scores = torch.softmax(logits, dim=-1)
        proto_summary = torch.matmul(scores, self.prototypes)
        return {
            "prototype_logits": logits,
            "prototype_scores": scores,
            "proto_summary": proto_summary,
        }


class EvidenceSelector(nn.Module):
    def __init__(self, units: int, num_evidence: int = 3, span_radius: int = 2):
        super().__init__()
        self.num_evidence = num_evidence
        self.init_sigma = max(1.0, float(span_radius))
        self.min_sigma_scale = 0.08
        self.max_sigma_scale = 0.45
        if num_evidence == 3:
            sigma_priors = torch.tensor([0.10, 0.20, 0.34], dtype=torch.float32)
        else:
            sigma_priors = torch.linspace(0.10, 0.34, steps=num_evidence, dtype=torch.float32)
        self.register_buffer("sigma_priors", sigma_priors)
        self.score_proj = nn.Linear(units, units)
        self.slot_queries = nn.Parameter(torch.randn(num_evidence, units))
        self.mu_head = nn.Linear(units, 1)
        self.sigma_head = nn.Linear(units, 1)
        self.slot_bias = nn.Linear(units, 1)
        self.gate = nn.Sequential(
            nn.Linear(units * 2, units),
            nn.GELU(),
            nn.Linear(units, 1),
        )
        nn.init.xavier_uniform_(self.slot_queries)

    def forward(
        self,
        h: torch.Tensor,
        valid_mask: torch.Tensor,
        centers: torch.Tensor,
        context_vec: torch.Tensor,
        attn_weights: torch.Tensor,
    ):
        bsz, seq_len, units = h.shape
        attn_mean = attn_weights.mean(dim=1)
        enriched = h + self.score_proj(h)

        idx = torch.arange(seq_len, device=h.device).view(1, 1, seq_len).float()
        valid_f = valid_mask.float()
        lengths = valid_mask.sum(dim=1, keepdim=True).clamp_min(1).float()
        rel_pos = (idx.squeeze(1) - centers.view(-1, 1).float()) / lengths

        slot_cond = context_vec.unsqueeze(1) + self.slot_queries.unsqueeze(0)
        mu = torch.tanh(self.mu_head(slot_cond)).squeeze(-1)
        sigma_delta = 0.06 * torch.tanh(self.sigma_head(slot_cond)).squeeze(-1)
        sigma = self.sigma_priors.unsqueeze(0) + sigma_delta + (0.5 * self.init_sigma / lengths)
        sigma = sigma.clamp(min=self.min_sigma_scale, max=self.max_sigma_scale)

        gaussian = -0.5 * ((rel_pos.unsqueeze(1) - mu.unsqueeze(-1)) / sigma.unsqueeze(-1).clamp_min(1e-4)) ** 2
        attn_prior = attn_mean.unsqueeze(1).clamp_min(1e-8).log()
        slot_logits = gaussian + attn_prior + self.slot_bias(slot_cond).squeeze(-1).unsqueeze(-1)
        slot_logits = slot_logits.masked_fill(~valid_mask.unsqueeze(1), float("-inf"))

        span_weights = torch.softmax(slot_logits, dim=-1)
        span_weights = torch.nan_to_num(span_weights, nan=0.0, posinf=0.0, neginf=0.0)

        evidence_vectors = torch.einsum("bkn,bnu->bku", span_weights, enriched)
        expected_positions = torch.sum(span_weights * idx, dim=-1)

        context_expand = context_vec.unsqueeze(1).expand(-1, self.num_evidence, -1)
        gate_input = torch.cat([evidence_vectors, context_expand], dim=-1)
        evidence_logits = self.gate(gate_input).squeeze(-1)
        evidence_weights = torch.softmax(evidence_logits, dim=-1)
        evidence_summary = torch.sum(evidence_vectors * evidence_weights.unsqueeze(-1), dim=1)

        return {
            "evidence_positions": expected_positions,
            "evidence_vectors": evidence_vectors,
            "evidence_weights": evidence_weights,
            "evidence_summary": evidence_summary,
            "evidence_mu": mu,
            "evidence_sigma": sigma,
            "evidence_sigma_prior": self.sigma_priors.unsqueeze(0).expand(bsz, -1),
            "span_weights": span_weights,
        }


class PTMModelV2(nn.Module):
    def __init__(
        self,
        units: int = 256,
        heads: int = 4,
        num_buckets: int = 16,
        max_distance: int = 32,
        num_prototypes: int = 12,
        num_evidence: int = 3,
        span_radius: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = SiteRelativeEncoder(units=units, num_buckets=num_buckets, max_distance=max_distance)
        self.center_attn = CenterConditionedAttention(
            units=units,
            heads=heads,
            num_buckets=num_buckets,
            dropout=dropout,
        )
        self.prototype_bank = MotifPrototypeBank(units=units, num_prototypes=num_prototypes)
        self.evidence_selector = EvidenceSelector(
            units=units,
            num_evidence=num_evidence,
            span_radius=span_radius,
        )
        self.classifier = nn.Sequential(
            nn.Linear(units * 4, units),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(units, 1),
        )

    def forward(self, x_emb: torch.Tensor, x_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        enc = self.encoder(x_emb, x_mask)
        attn_out = self.center_attn(enc["h"], enc["valid_mask"], enc["centers"], enc["bucket"])
        proto_out = self.prototype_bank(attn_out["context_vec"])
        evidence_out = self.evidence_selector(
            enc["h"],
            enc["valid_mask"],
            enc["centers"],
            attn_out["context_vec"],
            attn_out["attn_weights"],
        )

        fused = torch.cat(
            [
                attn_out["center_repr"],
                attn_out["context_vec"],
                proto_out["proto_summary"],
                evidence_out["evidence_summary"],
            ],
            dim=-1,
        )
        logits = self.classifier(fused).squeeze(-1)
        prob = torch.sigmoid(torch.clamp(logits, -30, 30))

        return {
            "prob": prob,
            "logits": logits,
            "centers": enc["centers"],
            "rel_pos": enc["rel_pos"],
            "attn_weights": attn_out["attn_weights"],
            "prototype_logits": proto_out["prototype_logits"],
            "prototype_scores": proto_out["prototype_scores"],
            "evidence_positions": evidence_out["evidence_positions"],
            "evidence_weights": evidence_out["evidence_weights"],
            "evidence_mu": evidence_out["evidence_mu"],
            "evidence_sigma": evidence_out["evidence_sigma"],
            "evidence_sigma_prior": evidence_out["evidence_sigma_prior"],
            "span_weights": evidence_out["span_weights"],
        }
