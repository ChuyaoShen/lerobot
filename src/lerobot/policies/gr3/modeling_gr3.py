#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
GR-3 Policy Implementation.

GR-3 (arXiv:2507.15493) is a Vision-Language-Action (VLA) model:
- VLM backbone: Qwen2.5-VL-3B-Instruct for encoding images + language
- Action head: Flow-matching Diffusion Transformer (DiT) with cross-attention
  to VLM hidden states
- Training: Flow matching objective (velocity prediction)
- Inference: Euler integration from noise to action (5 steps, Δτ=0.2)

Reference: starVLA/QwenGR00T for the Qwen-VL + DiT action head pattern.
"""

import logging
import math
from collections import deque
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from torch.distributions import Beta

from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)
from lerobot.utils.import_utils import _transformers_available, require_package

from ..pretrained import PreTrainedPolicy
from .configuration_gr3 import GR3Config

if TYPE_CHECKING or _transformers_available:
    from transformers import AutoProcessor

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
#  Building blocks
# ──────────────────────────────────────────────────────────────────────


def swish(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for timestep embedding."""

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps: Tensor) -> Tensor:
        """
        Args:
            timesteps: shape (B, T) or (B,)
        Returns:
            Encoding of shape (B, T, embedding_dim) or (B, embedding_dim)
        """
        squeeze = False
        if timesteps.dim() == 1:
            timesteps = timesteps.unsqueeze(1)
            squeeze = True

        timesteps = timesteps.float()
        half_dim = self.embedding_dim // 2
        exponent = -torch.arange(half_dim, dtype=torch.float, device=timesteps.device) * (
            math.log(10000.0) / half_dim
        )
        freqs = timesteps.unsqueeze(-1) * exponent.exp()
        enc = torch.cat([torch.sin(freqs), torch.cos(freqs)], dim=-1)

        if squeeze:
            enc = enc.squeeze(1)
        return enc


class TimestepEncoder(nn.Module):
    """Encode continuous timesteps into embedding vectors via sinusoidal + MLP."""

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.sinusoidal = SinusoidalPositionalEncoding(256)
        self.mlp = nn.Sequential(
            nn.Linear(256, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(self, timesteps: Tensor) -> Tensor:
        """
        Args:
            timesteps: shape (B,) integer timesteps
        Returns:
            shape (B, embedding_dim)
        """
        proj = self.sinusoidal(timesteps.float())
        return self.mlp(proj)


class AdaLayerNorm(nn.Module):
    """Adaptive Layer Norm conditioned on timestep embedding (AdaLN)."""

    def __init__(self, embedding_dim: int, eps: float = 1e-5):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, 2 * embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim, eps=eps, elementwise_affine=False)

    def forward(self, x: Tensor, temb: Tensor) -> Tensor:
        """
        Args:
            x: (B, T, D) hidden states
            temb: (B, D) timestep embedding
        """
        params = self.linear(self.silu(temb))
        scale, shift = params.chunk(2, dim=-1)
        return self.norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class GR3ActionEncoder(nn.Module):
    """Encode noisy actions + flow-matching timestep into embeddings.

    Following the GR-3 / GR00T pattern:
    - W1: action_dim → hidden_size
    - sinusoidal encoding of timestep
    - W2: concat(action_emb, time_emb) → hidden_size with swish
    - W3: hidden_size → hidden_size
    """

    def __init__(self, action_dim: int, hidden_size: int):
        super().__init__()
        self.w1 = nn.Linear(action_dim, hidden_size)
        self.w2 = nn.Linear(2 * hidden_size, hidden_size)
        self.w3 = nn.Linear(hidden_size, hidden_size)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions: Tensor, timesteps: Tensor) -> Tensor:
        """
        Args:
            actions: (B, T, action_dim) noisy action trajectory
            timesteps: (B,) discretized timestep indices
        Returns:
            (B, T, hidden_size)
        """
        b, t, _ = actions.shape
        # Expand scalar timestep across the temporal dimension
        timesteps_expanded = timesteps.unsqueeze(1).expand(-1, t)

        a_emb = self.w1(actions)
        tau_emb = self.pos_encoding(timesteps_expanded).to(dtype=a_emb.dtype)
        x = swish(self.w2(torch.cat([a_emb, tau_emb], dim=-1)))
        return self.w3(x)


# ──────────────────────────────────────────────────────────────────────
#  DiT Transformer block + full model
# ──────────────────────────────────────────────────────────────────────


class GR3TransformerBlock(nn.Module):
    """A single DiT transformer block with optional cross-attention.

    Key design from GR-3 paper: RMSNorm after linear layers in attention
    and FFN for training stability and better instruction following.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: int,
        cross_attention_dim: int | None = None,
        dropout: float = 0.0,
        final_dropout: bool = False,
        norm_type: str = "ada_norm",
    ):
        super().__init__()
        self.dim = dim
        self.norm_type = norm_type

        # Pre-attention norm
        if norm_type == "ada_norm":
            self.norm1 = AdaLayerNorm(dim)
        else:
            self.norm1 = nn.LayerNorm(dim)

        # Attention (self or cross depending on whether cross_attention_dim is set)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
            kdim=cross_attention_dim if cross_attention_dim else dim,
            vdim=cross_attention_dim if cross_attention_dim else dim,
        )
        # RMSNorm after attention output projection (GR-3 paper key design)
        self.attn_post_norm = nn.RMSNorm(dim)

        # FFN
        self.norm3 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(4 * dim, dim),
        )
        # RMSNorm after FFN (GR-3 paper key design)
        self.ff_post_norm = nn.RMSNorm(dim)

        self.final_dropout = nn.Dropout(dropout) if final_dropout and dropout > 0 else None
        self.is_cross_attention = cross_attention_dim is not None

    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor | None = None,
        encoder_attention_mask: Tensor | None = None,
        temb: Tensor | None = None,
    ) -> Tensor:
        # Pre-norm
        if self.norm_type == "ada_norm" and temb is not None:
            norm_hidden = self.norm1(hidden_states, temb)
        else:
            norm_hidden = self.norm1(hidden_states)

        # Attention
        if self.is_cross_attention and encoder_hidden_states is not None:
            attn_out, _ = self.attn(
                query=norm_hidden,
                key=encoder_hidden_states,
                value=encoder_hidden_states,
                key_padding_mask=encoder_attention_mask,
            )
        else:
            attn_out, _ = self.attn(
                query=norm_hidden,
                key=norm_hidden,
                value=norm_hidden,
            )

        attn_out = self.attn_post_norm(attn_out)
        if self.final_dropout is not None:
            attn_out = self.final_dropout(attn_out)
        hidden_states = hidden_states + attn_out

        # FFN
        norm_hidden = self.norm3(hidden_states)
        ff_out = self.ff(norm_hidden)
        ff_out = self.ff_post_norm(ff_out)
        hidden_states = hidden_states + ff_out

        return hidden_states


class GR3DiT(nn.Module):
    """Diffusion Transformer for GR-3 action prediction.

    Architecture:
    - Interleaved cross-attention (even layers) and self-attention (odd layers)
    - AdaLN for timestep conditioning
    - Cross-attention to VLM hidden states
    - Output projection with AdaLN-conditioned scale/shift
    """

    def __init__(self, config: GR3Config):
        super().__init__()
        inner_dim = config.dit_inner_dim
        self.inner_dim = inner_dim

        # Timestep encoder
        self.timestep_encoder = TimestepEncoder(embedding_dim=inner_dim)

        # Transformer blocks: interleave cross-attn and self-attn
        blocks = []
        for idx in range(config.dit_num_layers):
            use_self_attn = idx % 2 == 1 and config.dit_interleave_self_attention
            cross_dim = config.vlm_hidden_size if not use_self_attn else None

            blocks.append(
                GR3TransformerBlock(
                    dim=inner_dim,
                    num_heads=config.dit_num_heads,
                    head_dim=config.dit_head_dim,
                    cross_attention_dim=cross_dim,
                    dropout=config.dit_dropout,
                    final_dropout=config.dit_final_dropout,
                    norm_type="ada_norm",
                )
            )
        self.transformer_blocks = nn.ModuleList(blocks)

        # Output head with AdaLN conditioning
        self.norm_out = nn.LayerNorm(inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out_1 = nn.Linear(inner_dim, 2 * inner_dim)
        self.proj_out_2 = nn.Linear(inner_dim, config.dit_output_dim)

        self.config = config

    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: list[Tensor],
        timestep: Tensor,
        encoder_attention_mask: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            hidden_states: (B, T_action, inner_dim) - action/state sequence
            encoder_hidden_states: list of (B, T_vlm, vlm_hidden_size) - per-layer
                VLM hidden states from the latter half, one per DiT layer.
                Cross-attention layers use their paired VLM layer; self-attention
                layers ignore theirs.
            timestep: (B,) - discretized timesteps
            encoder_attention_mask: (B, T_vlm) - attention mask for VLM tokens
        Returns:
            (B, T_action, output_dim)
        """
        temb = self.timestep_encoder(timestep)

        for idx, block in enumerate(self.transformer_blocks):
            use_self_attn = idx % 2 == 1 and self.config.dit_interleave_self_attention
            if use_self_attn:
                hidden_states = block(hidden_states, temb=temb)
            else:
                hidden_states = block(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states[idx],
                    encoder_attention_mask=encoder_attention_mask,
                    temb=temb,
                )

        # Output projection with AdaLN
        shift, scale = self.proj_out_1(F.silu(temb)).chunk(2, dim=-1)
        hidden_states = self.norm_out(hidden_states) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return self.proj_out_2(hidden_states)


# ──────────────────────────────────────────────────────────────────────
#  GR3Model: Qwen2.5-VL + DiT
# ──────────────────────────────────────────────────────────────────────


class GR3Model(nn.Module):
    """Full GR-3 model combining Qwen2.5-VL backbone with DiT action head.

    Training: flow matching loss on velocity field.
    Inference: Euler integration from noise → action over num_inference_steps.
    """

    def __init__(self, config: GR3Config):
        super().__init__()
        self.config = config

        inner_dim = config.dit_inner_dim

        # ── VLM backbone ──
        from transformers import Qwen2_5_VLForConditionalGeneration

        self.vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            config.vlm_model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )

        # Apply freeze settings
        if config.freeze_vlm:
            for param in self.vlm.parameters():
                param.requires_grad = False
        elif config.freeze_vision_encoder:
            for param in self.vlm.visual.parameters():
                param.requires_grad = False

        # ── State encoder ──
        self.state_encoder = nn.Sequential(
            nn.Linear(config.max_state_dim, config.dit_output_dim),
            nn.ReLU(),
            nn.Linear(config.dit_output_dim, inner_dim),
        )

        # ── Action encoder ──
        self.action_encoder = GR3ActionEncoder(
            action_dim=config.max_action_dim,
            hidden_size=inner_dim,
        )

        # ── Action decoder ──
        self.action_decoder = nn.Sequential(
            nn.Linear(config.dit_output_dim, config.dit_output_dim),
            nn.ReLU(),
            nn.Linear(config.dit_output_dim, config.max_action_dim),
        )

        # ── Learnable future tokens ──
        self.future_tokens = nn.Embedding(config.num_future_tokens, inner_dim)
        nn.init.normal_(self.future_tokens.weight, mean=0.0, std=0.02)

        # ── Positional embedding for action sequence ──
        max_seq = config.chunk_size + config.num_future_tokens + 1  # +1 for state
        self.position_embedding = nn.Embedding(max_seq, inner_dim)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        # ── DiT action head ──
        self.dit = GR3DiT(config)

        # ── Flow matching noise distribution ──
        self.beta_dist = Beta(config.time_beta_alpha, config.time_beta_beta)

        # ── VLM processor for image/text processing ──
        self.processor = AutoProcessor.from_pretrained(config.vlm_model_name)

    def _sample_time(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        """Sample flow matching timesteps from Beta distribution."""
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype).clamp(max=self.config.time_scale)
        return (self.config.time_scale - sample) / self.config.time_scale

    def _encode_vlm(
        self,
        pixel_values: Tensor | None,
        image_grid_thw: Tensor | None,
        lang_tokens: Tensor,
        lang_masks: Tensor,
    ) -> list[Tensor]:
        """Encode images + language through Qwen2.5-VL backbone.

        Returns hidden states from the latter half of VLM layers for
        layerwise cross-attention in the DiT. Each DiT layer i is paired
        1-to-1 with VLM latter-half layer i (per the GR-3 paper: "the action
        DiT utilizes only the KV cache from the latter half of the VLM layers").
        """
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.vlm(
                input_ids=lang_tokens,
                attention_mask=lang_masks,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                output_hidden_states=True,
                return_dict=True,
            )
            # outputs.hidden_states is (embedding_output, layer_0, ..., layer_N-1)
            all_hidden = outputs.hidden_states
            num_vlm_layers = len(all_hidden) - 1  # exclude embedding output
            latter_half_start = num_vlm_layers // 2
            # all_hidden[i+1] = output of VLM layer i (0-indexed)
            latter_half = list(all_hidden[latter_half_start + 1 :])

        if len(latter_half) != self.config.dit_num_layers:
            raise ValueError(
                f"Expected {self.config.dit_num_layers} VLM latter-half hidden states "
                f"(dit_num_layers), but got {len(latter_half)} "
                f"(VLM has {num_vlm_layers} layers, latter half = {num_vlm_layers - latter_half_start}). "
                f"Ensure dit_num_layers == num_vlm_layers // 2."
            )

        return latter_half

    def forward(
        self,
        pixel_values: Tensor | None,
        image_grid_thw: Tensor | None,
        lang_tokens: Tensor,
        lang_masks: Tensor,
        state: Tensor,
        actions: Tensor,
    ) -> Tensor:
        """Training forward pass with flow matching loss.

        Args:
            pixel_values: (total_patches, C*tp*p*p) from image processor, or None
            image_grid_thw: (num_images, 3) from image processor, or None
            lang_tokens: (B, seq_len) tokenized language
            lang_masks: (B, seq_len) attention masks
            state: (B, max_state_dim) robot state
            actions: (B, chunk_size, max_action_dim) target actions

        Returns:
            (B, chunk_size, max_action_dim) per-element MSE losses
        """
        device = actions.device

        # ── Encode VLM features (shared across diffusion repeats) ──
        vlm_features = self._encode_vlm(pixel_values, image_grid_thw, lang_tokens, lang_masks)

        # ── Repeat for multiple diffusion timestep samples ──
        num_repeats = self.config.repeated_diffusion_steps
        actions_rep = actions.repeat(num_repeats, 1, 1)
        vlm_features_rep = [f.repeat(num_repeats, 1, 1) for f in vlm_features]
        state_rep = state.repeat(num_repeats, 1)
        b_rep = actions_rep.shape[0]

        # ── Sample noise and time ──
        noise = torch.randn_like(actions_rep)
        t = self._sample_time(b_rep, device=device, dtype=actions_rep.dtype)
        t_expanded = t[:, None, None]

        # ── Interpolate noisy trajectory ──
        noisy_actions = (1 - t_expanded) * noise + t_expanded * actions_rep
        velocity_target = actions_rep - noise

        # ── Discretize time for encoder ──
        t_discrete = (t * self.config.num_timestep_buckets).long()

        # ── Encode action trajectory ──
        action_features = self.action_encoder(noisy_actions, t_discrete)

        # ── Encode state ──
        state_features = self.state_encoder(state_rep).unsqueeze(1)  # (B, 1, inner_dim)

        # ── Assemble DiT input: [state, future_tokens, action_features] ──
        future_tok = self.future_tokens.weight.unsqueeze(0).expand(b_rep, -1, -1)
        sa_embs = torch.cat([state_features, future_tok, action_features], dim=1)

        # ── Add positional embeddings ──
        pos_ids = torch.arange(sa_embs.shape[1], device=device, dtype=torch.long)
        sa_embs = sa_embs + self.position_embedding(pos_ids).unsqueeze(0)

        # ── DiT forward ──
        with torch.autocast("cuda", dtype=torch.float32):
            dit_output = self.dit(
                hidden_states=sa_embs,
                encoder_hidden_states=vlm_features_rep,
                timestep=t_discrete,
            )

        # ── Decode predicted velocity (only action portion) ──
        pred_velocity = self.action_decoder(dit_output[:, -actions_rep.shape[1] :])

        # ── Per-element MSE loss ──
        losses = (pred_velocity - velocity_target) ** 2

        # ── Average across repeated diffusion steps ──
        b_orig = actions.shape[0]
        losses = losses.view(num_repeats, b_orig, *losses.shape[1:]).mean(dim=0)

        return losses

    @torch.no_grad()
    def sample_actions(
        self,
        pixel_values: Tensor | None,
        image_grid_thw: Tensor | None,
        lang_tokens: Tensor,
        lang_masks: Tensor,
        state: Tensor,
    ) -> Tensor:
        """Inference: Euler integration from noise to action.

        Args:
            pixel_values: (total_patches, C*tp*p*p) from image processor, or None
            image_grid_thw: (num_images, 3) from image processor, or None
            lang_tokens, lang_masks: tokenized language inputs
            state: (B, max_state_dim) robot state

        Returns:
            (B, chunk_size, max_action_dim) predicted action chunk
        """
        device = state.device
        batch_size = state.shape[0]

        # ── Encode VLM features once ──
        vlm_features = self._encode_vlm(pixel_values, image_grid_thw, lang_tokens, lang_masks)

        # ── Initialize from noise ──
        actions = torch.randn(
            batch_size,
            self.config.chunk_size,
            self.config.max_action_dim,
            device=device,
            dtype=vlm_features[0].dtype,
        )

        # ── State features ──
        state_features = self.state_encoder(state).unsqueeze(1)

        # ── Euler integration ──
        num_steps = self.config.num_inference_steps
        dt = 1.0 / num_steps

        for step in range(num_steps):
            t_cont = step / float(num_steps)
            t_discrete_val = int(t_cont * self.config.num_timestep_buckets)
            t_discrete = torch.full((batch_size,), t_discrete_val, device=device, dtype=torch.long)

            # Encode current noisy actions
            action_features = self.action_encoder(actions, t_discrete)

            # Assemble input
            future_tok = self.future_tokens.weight.unsqueeze(0).expand(batch_size, -1, -1)
            sa_embs = torch.cat([state_features, future_tok, action_features], dim=1)
            pos_ids = torch.arange(sa_embs.shape[1], device=device, dtype=torch.long)
            sa_embs = sa_embs + self.position_embedding(pos_ids).unsqueeze(0)

            # DiT forward
            with torch.autocast("cuda", dtype=torch.float32):
                dit_output = self.dit(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vlm_features,
                    timestep=t_discrete,
                )

            # Decode velocity
            pred_velocity = self.action_decoder(dit_output[:, -self.config.chunk_size :])

            # Euler step
            actions = actions + dt * pred_velocity

        return actions


# ──────────────────────────────────────────────────────────────────────
#  GR3Policy: LeRobot wrapper
# ──────────────────────────────────────────────────────────────────────


class GR3Policy(PreTrainedPolicy):
    """GR-3 Policy for LeRobot.

    Usage:
        lerobot-train --policy.type=gr3 ...
    """

    config_class = GR3Config
    name = "gr3"

    def __init__(self, config: GR3Config, **kwargs):
        require_package("transformers", extra="gr3")
        super().__init__(config)
        config.validate_features()
        self.config = config

        self.model = GR3Model(config)

        if config.gradient_checkpointing:
            self.model.vlm.gradient_checkpointing_enable()

        self.model.to(config.device)
        self.reset()

    def get_optim_params(self) -> dict:
        return self.parameters()

    def reset(self):
        """Reset internal state - called when environment resets."""
        self._action_queue = deque(maxlen=self.config.n_action_steps)

    # ── Image preprocessing ──

    def _preprocess_images(self, batch: dict[str, Tensor]) -> tuple[Tensor | None, Tensor | None]:
        """Preprocess images for Qwen2.5-VL using the built-in image processor.

        Collects images from the batch, resizes to target resolution, then
        delegates CLIP normalization, 3D-patch conversion, and grid_thw
        computation to ``Qwen2VLImageProcessor``.

        Images are ordered batch-major (all cameras for batch item 0, then
        batch item 1, …) to match the ``<|image_pad|>`` token order in
        ``input_ids``.

        Returns:
            pixel_values: ``(total_patches, C*tp*p*p)`` ready for the VLM, or *None*.
            image_grid_thw: ``(num_images, 3)`` ready for the VLM, or *None*.
        """
        device = next(self.parameters()).device
        image_processor = self.model.processor.image_processor

        present_keys = [k for k in self.config.image_features if k in batch]

        if not present_keys:
            raise ValueError(
                f"No image features found in batch. Expected at least one of: "
                f"{list(self.config.image_features)}"
            )

        batch_size = batch[present_keys[0]].shape[0]
        all_images: list[Tensor] = []

        for b in range(batch_size):
            for key in present_keys:
                img = batch[key][b].to(device=device, dtype=torch.float32)  # (C, H, W)

                # Ensure (C, H, W)
                if img.shape[0] != 3 and img.shape[-1] == 3:
                    img = img.permute(2, 0, 1)

                # Resize to target resolution
                if tuple(img.shape[1:]) != tuple(self.config.image_resolution):
                    img = F.interpolate(
                        img.unsqueeze(0),
                        size=self.config.image_resolution,
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0)

                all_images.append(img)

        if not all_images:
            return None, None

        # Delegate normalization + 3D-patch conversion + grid computation
        # to the Qwen2.5-VL image processor. Our images are already [0, 1]
        # float tensors at the target resolution, so skip resize and rescale.
        result = image_processor(
            images=all_images,
            do_resize=False,
            do_rescale=False,
            return_tensors="pt",
        )

        pixel_values = result["pixel_values"].to(device=device)
        image_grid_thw = result["image_grid_thw"].to(device=device)

        return pixel_values, image_grid_thw

    def _prepare_state(self, batch: dict[str, Tensor]) -> Tensor:
        """Pad state to max_state_dim."""
        state = batch[OBS_STATE]
        if state.shape[-1] < self.config.max_state_dim:
            pad_size = self.config.max_state_dim - state.shape[-1]
            state = F.pad(state, (0, pad_size))
        return state

    def _prepare_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Pad action to max_action_dim."""
        actions = batch[ACTION]
        if actions.shape[-1] < self.config.max_action_dim:
            pad_size = self.config.max_action_dim - actions.shape[-1]
            actions = F.pad(actions, (0, pad_size))
        return actions

    # ── Core policy methods ──

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations."""
        self.eval()

        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))

        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Predict a full chunk of actions."""
        self.eval()

        pixel_values, image_grid_thw = self._preprocess_images(batch)
        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        state = self._prepare_state(batch)

        actions = self.model.sample_actions(pixel_values, image_grid_thw, lang_tokens, lang_masks, state)

        # Unpad to actual action dim
        original_action_dim = self.config.output_features[ACTION].shape[0]
        actions = actions[:, :, :original_action_dim]

        return actions

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Training forward pass.

        Returns:
            (loss, info_dict) where loss is a scalar tensor.
        """
        pixel_values, image_grid_thw = self._preprocess_images(batch)
        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        state = self._prepare_state(batch)
        actions = self._prepare_action(batch)

        # Forward with flow matching loss
        losses = self.model.forward(pixel_values, image_grid_thw, lang_tokens, lang_masks, state, actions)

        # Truncate to actual action dims
        original_action_dim = self.config.output_features[ACTION].shape[0]
        losses = losses[:, :, :original_action_dim]

        loss = losses.mean()
        loss_dict = {
            "loss": loss.item(),
            "loss_per_dim": losses.mean(dim=[0, 1]).detach().cpu().numpy().tolist(),
        }

        return loss, loss_dict

    def _get_default_peft_targets(self) -> dict[str, any]:
        """Return default PEFT target modules for GR-3 fine-tuning."""
        return {
            "target_modules": r"(.*\.dit\..*\.attn\..*|model\.state_encoder|model\.action_decoder|model\.action_encoder)",
            "modules_to_save": [],
        }
