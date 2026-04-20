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

"""Pre- and post-processor pipelines for the GR-3 policy."""

from typing import Any

import torch

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    ComplementaryDataProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RenameObservationsProcessorStep,
    TokenizerProcessorStep,
    UnnormalizerProcessorStep,
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME

from .configuration_gr3 import GR3Config


@ProcessorStepRegistry.register(name="gr3_language_processor")
class GR3LanguageProcessor(ComplementaryDataProcessorStep):
    """Formats task descriptions as Qwen2.5-VL chat prompts with image placeholders.

    Wraps each task string in the Qwen2.5-VL chat template, inserting the
    correct number of ``<|image_pad|>`` tokens per camera so that the VLM
    can inject vision features at the right positions in ``input_ids``.
    """

    def __init__(self, num_cameras: int = 1, num_image_tokens_per_image: int = 196):
        self.num_cameras = num_cameras
        self.num_image_tokens_per_image = num_image_tokens_per_image

    def _format_prompt(self, task: str) -> str:
        """Build a Qwen2.5-VL chat prompt with image pad tokens."""
        image_placeholder = (
            "<|vision_start|>"
            + "<|image_pad|>" * self.num_image_tokens_per_image
            + "<|vision_end|>"
        )
        image_section = "".join(image_placeholder for _ in range(self.num_cameras))
        return (
            f"<|im_start|>user\n"
            f"{image_section}"
            f"{task}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    def complementary_data(self, complementary_data):
        if "task" not in complementary_data:
            return complementary_data

        task = complementary_data["task"]
        if task is None:
            return complementary_data

        new_data = dict(complementary_data)
        if isinstance(task, str):
            new_data["task"] = self._format_prompt(task)
        elif isinstance(task, list) and all(isinstance(t, str) for t in task):
            new_data["task"] = [self._format_prompt(t) for t in task]

        return new_data

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


def make_gr3_pre_post_processors(
    config: GR3Config,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Construct pre-processor and post-processor pipelines for the GR-3 policy.

    Pre-processing:
    1. Rename observations (for pretrained compatibility)
    2. Add batch dimension
    3. Format language for Qwen2.5-VL
    4. Tokenize text with Qwen2.5-VL tokenizer
    5. Move to device
    6. Normalize state/actions

    Post-processing:
    1. Unnormalize actions
    2. Move to CPU
    """
    input_steps: list[ProcessorStep] = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        GR3LanguageProcessor(
            num_cameras=len(config.image_features),
            num_image_tokens_per_image=config.num_image_tokens_per_image,
        ),
        TokenizerProcessorStep(
            tokenizer_name=config.vlm_model_name,
            max_length=config.tokenizer_max_length,
            padding_side="right",
            padding="max_length",
        ),
        DeviceProcessorStep(device=config.device),
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
    ]

    output_steps: list[ProcessorStep] = [
        UnnormalizerProcessorStep(
            features=config.output_features,
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        DeviceProcessorStep(device="cpu"),
    ]

    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
