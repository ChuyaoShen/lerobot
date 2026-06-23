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
GR-3 Policy Configuration.

GR-3 (arXiv:2507.15493) is a Vision-Language-Action (VLA) model that uses
Qwen2.5-VL as the VLM backbone and a flow-matching Diffusion Transformer (DiT)
for action prediction.
"""

from dataclasses import dataclass, field

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature, PreTrainedConfig
from lerobot.optim import AdamWConfig, CosineDecayWithWarmupSchedulerConfig
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

DEFAULT_IMAGE_SIZE = 392  # Must be a multiple of patch_size(14) * spatial_merge_size(2) = 28


@PreTrainedConfig.register_subclass("gr3")
@dataclass
class GR3Config(PreTrainedConfig):
    """Configuration for the GR-3 policy.

    GR-3 uses Qwen2.5-VL-3B-Instruct as the vision-language backbone and
    a cross-attention Diffusion Transformer (DiT) with flow matching for
    action prediction.
    """

    # ── VLM backbone ──
    vlm_model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    vlm_hidden_size: int = 2048  # Qwen2.5-VL-3B hidden size
    freeze_vlm: bool = False
    freeze_vision_encoder: bool = False

    # ── DiT action head ──
    dit_num_layers: int = 16
    dit_num_heads: int = 12
    dit_head_dim: int = 64
    dit_dropout: float = 0.2
    dit_final_dropout: bool = True
    dit_output_dim: int = 1024
    # Interleave self-attention (odd layers) with cross-attention (even layers)
    dit_interleave_self_attention: bool = True
    # Number of learnable future tokens prepended to action sequence
    num_future_tokens: int = 32

    # ── Action ──
    n_obs_steps: int = 1
    chunk_size: int = 16  # Number of action steps to predict (k in the paper)
    n_action_steps: int = 16  # Number of action steps to execute

    # Shorter state and action vectors will be padded to these dimensions
    max_state_dim: int = 32
    max_action_dim: int = 32

    # ── Flow matching ──
    num_inference_steps: int = 5  # Paper uses Δτ=0.2 → 5 steps
    time_beta_alpha: float = 1.5  # Beta distribution alpha for time sampling
    time_beta_beta: float = 1.0  # Beta distribution beta for time sampling
    time_scale: float = 0.999  # Max time value (noise_s in starVLA)
    num_timestep_buckets: int = 1000  # Discretization buckets for timestep encoding

    # Number of flow matching timestep samples per VLM forward pass (for training efficiency)
    repeated_diffusion_steps: int = 4

    # ── Image ──
    image_resolution: tuple[int, int] = (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)
    empty_cameras: int = 0

    # ── Normalization ──
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # ── Training ──
    gradient_checkpointing: bool = False
    compile_model: bool = False
    compile_mode: str = "max-autotune"
    device: str | None = None

    # ── Optimizer ──
    optimizer_lr: float = 2.5e-5
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.01
    optimizer_grad_clip_norm: float = 1.0

    # ── Scheduler ──
    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    # ── Tokenizer ──
    tokenizer_max_length: int = 1024  # Must accommodate image pad tokens + task text

    # ── VLM vision constants (Qwen2.5-VL defaults) ──
    vision_patch_size: int = 14
    vision_merge_size: int = 2

    @property
    def num_image_tokens_per_image(self) -> int:
        """Number of ``<|image_pad|>`` tokens per image for Qwen2.5-VL.

        Derived from image_resolution, patch_size, and merge_size:
        ``grid_h * grid_w / merge_size²`` (grid_t = 1 for static images).
        """
        H, W = self.image_resolution
        grid_h = H // self.vision_patch_size
        grid_w = W // self.vision_patch_size
        return (grid_h * grid_w) // (self.vision_merge_size**2)

    def __post_init__(self):
        super().__post_init__()

        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) cannot be greater than chunk_size ({self.chunk_size})"
            )

    def validate_features(self) -> None:
        """Validate and set up input/output features."""
        for i in range(self.empty_cameras):
            key = f"{OBS_IMAGES}.empty_camera_{i}"
            empty_camera = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, *self.image_resolution),
            )
            self.input_features[key] = empty_camera

        if OBS_STATE not in self.input_features:
            state_feature = PolicyFeature(
                type=FeatureType.STATE,
                shape=(self.max_state_dim,),
            )
            self.input_features[OBS_STATE] = state_feature

        if ACTION not in self.output_features:
            action_feature = PolicyFeature(
                type=FeatureType.ACTION,
                shape=(self.max_action_dim,),
            )
            self.output_features[ACTION] = action_feature

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self):
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None

    @property
    def dit_inner_dim(self) -> int:
        """The inner dimension of the DiT (num_heads * head_dim)."""
        return self.dit_num_heads * self.dit_head_dim
