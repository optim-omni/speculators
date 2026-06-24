from typing import Any, Literal

from pydantic import Field, field_serializer, field_validator
from transformers import AutoConfig, PretrainedConfig
from transformers.models.llama.configuration_llama import LlamaConfig

from speculators import SpeculatorModelConfig

__all__ = ["Eagle1TrainSpeculatorConfig"]


@SpeculatorModelConfig.register("eagle1_train")
class Eagle1TrainSpeculatorConfig(SpeculatorModelConfig):
    """Configuration for the training-only EAGLE1-style drafter.

    The model consumes exactly one verifier auxiliary hidden-state layer plus the
    verifier final hidden state used to construct KL targets. It reuses the
    EAGLE3 training data and loss pipeline while keeping the draft structure to
    token embedding -> fusion FC -> decoder layer(s) -> LM head.
    """

    speculators_model_type: Literal["eagle1_train"] = "eagle1_train"
    architectures: list[str] = Field(
        default_factory=lambda: ["Eagle1TrainDraftModel"],
        description="Model architectures that can load these weights",
    )

    transformer_layer_config: PretrainedConfig = Field(
        default_factory=LlamaConfig,
        description="Configuration for the transformer decoder layer",
    )

    draft_vocab_size: int = Field(
        default=32000,
        description="Size of draft model vocabulary for speculation",
    )

    target_hidden_size: int | None = Field(
        default=None,
        description="Hidden size of the target model (if different from draft model)",
    )

    eagle_aux_hidden_state_layer_ids: list[int] | None = Field(
        default=None,
        description="Single verifier auxiliary hidden-state layer used by EAGLE1",
    )

    @field_serializer("transformer_layer_config")
    def serialize_transformer_config(self, value: PretrainedConfig) -> dict:
        return value.to_diff_dict()

    @field_validator("transformer_layer_config", mode="before")
    @classmethod
    def validate_transformer_config(cls, value: Any) -> PretrainedConfig:
        if isinstance(value, dict):
            config_class: type[PretrainedConfig] = LlamaConfig
            if "model_type" in value:
                config_class = AutoConfig.for_model(
                    model_type=value["model_type"]
                ).__class__
            return config_class(**value)
        return value

    @field_validator("eagle_aux_hidden_state_layer_ids")
    @classmethod
    def validate_single_aux_layer(cls, value: list[int] | None) -> list[int] | None:
        if value is not None and len(value) != 1:
            raise ValueError(
                "eagle1_train requires exactly one aux hidden-state layer, "
                f"got {value}."
            )
        return value

    @property
    def target_vocab_size(self) -> int:
        return self.transformer_layer_config.vocab_size  # type: ignore[return-value]
