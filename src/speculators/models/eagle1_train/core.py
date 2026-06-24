"""Training-only EAGLE1-style speculator model."""

import os
import warnings
from typing import Any, ClassVar

import torch
from torch import nn
from transformers import PretrainedConfig

from speculators.config import SpeculatorsConfig, VerifierConfig
from speculators.model import DraftVocabMixin, SpeculatorModel
from speculators.models.base_components import model_classes
from speculators.models.eagle1_train.config import Eagle1TrainSpeculatorConfig
from speculators.models.eagle3.metrics import compute_metrics
from speculators.models.metrics import kl_div_loss, resolve_loss_fn
from speculators.models.utils import get_verifier_config
from speculators.proposals.greedy import GreedyTokenProposalConfig

__all__ = ["Eagle1TrainDraftModel"]


DEFAULT_EAGLE1_TARGET_LAYER_IDS_WARNING = (
    "--target-layer-ids is not explicitly set for eagle1_train. Setting target "
    "layer to {target_layer_ids}. If vLLM datagen used a different aux layer, "
    "set --target-layer-ids explicitly."
)


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _resolve_eagle1_target_layer_ids(
    target_layer_ids: list[int] | None,
    verifier_name_or_path: str,
) -> list[int]:
    if target_layer_ids is not None:
        if len(target_layer_ids) != 1:
            raise ValueError(
                "eagle1_train requires exactly one target layer id, "
                f"got {target_layer_ids}."
            )
        return target_layer_ids

    num_layers = get_verifier_config(verifier_name_or_path).num_hidden_layers
    target_layer_ids = [max(0, num_layers - 3)]
    warnings.warn(
        DEFAULT_EAGLE1_TARGET_LAYER_IDS_WARNING.format(
            target_layer_ids=target_layer_ids
        ),
        stacklevel=3,
    )
    return target_layer_ids


def _packed_causal_mask(
    *,
    lengths: torch.Tensor | None,
    seq_len: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Build an additive causal mask that respects packed sample boundaries."""
    pos = torch.arange(seq_len, device=device)
    causal = pos[:, None] >= pos[None, :]

    if lengths is None:
        allowed = causal
    else:
        lengths = lengths.to(device=device, dtype=torch.long)
        ends = torch.cumsum(lengths, dim=0)
        segment_ids = torch.searchsorted(ends, pos, right=True)
        valid = pos < ends[-1].clamp(min=0)
        same_segment = (
            segment_ids[:, None] == segment_ids[None, :]
        ) & valid[:, None] & valid[None, :]
        padding_self = (~valid)[:, None] & (pos[:, None] == pos[None, :])
        allowed = (same_segment & causal) | padding_self

    mask = torch.zeros((1, 1, seq_len, seq_len), dtype=dtype, device=device)
    return mask.masked_fill(~allowed.unsqueeze(0).unsqueeze(0), torch.finfo(dtype).min)


@SpeculatorModel.register("eagle1_train")
class Eagle1TrainDraftModel(DraftVocabMixin, SpeculatorModel):
    """EAGLE1-style drafter trained through the EAGLE3 trainer path.

    The forward path takes one auxiliary verifier hidden state and current token
    ids, fuses them, runs a decoder layer, and computes the same KL/CE losses and
    metrics as EAGLE3 against logits derived from the verifier final hidden state.
    """

    config_class: ClassVar[type[Eagle1TrainSpeculatorConfig]] = (  # type: ignore[misc]
        Eagle1TrainSpeculatorConfig
    )
    _keys_to_ignore_on_load_missing: ClassVar[list[str]] = [  # type: ignore[misc]
        "embed_tokens.weight",
        "verifier_norm.weight",
        "verifier_lm_head.weight",
        "d2t",
        "t2d",
    ]
    _keys_to_ignore_on_save: ClassVar[list[str]] = [  # type: ignore[misc,assignment]
        "verifier_lm_head.weight",
        "verifier_norm.weight",
    ]

    t2d: torch.Tensor | None
    d2t: torch.Tensor | None

    def __init__(self, config: Eagle1TrainSpeculatorConfig) -> None:
        if config.transformer_layer_config._attn_implementation is None:  # noqa: SLF001
            config.transformer_layer_config._attn_implementation = "eager"  # noqa: SLF001
        config.transformer_layer_config.num_hidden_layers = 1

        super().__init__(config=config)
        self._init_vocab(config)

        tl_config = config.transformer_layer_config
        self._model_definitions = model_classes[tl_config.model_type]

        self.fusion_fc = nn.Linear(2 * self.hidden_size, self.hidden_size, bias=False)
        self.layers = nn.ModuleList(
            [self._model_definitions.decoder_layer_class(tl_config, layer_idx=0)]
        )
        self.rotary_emb = self._model_definitions.rotary_emb_class(tl_config)

        norm_class = self._model_definitions.norm_class
        self.norm = norm_class(self.hidden_size, eps=tl_config.rms_norm_eps)
        self.verifier_norm = norm_class(self.hidden_size, eps=tl_config.rms_norm_eps)
        self.verifier_norm.weight.requires_grad = False

        self._compiled_forward_impl = None
        if (
            _env_flag("MOE_SPEC_EAGLE1_TORCH_COMPILE")
            and torch.cuda.is_available()
            and hasattr(torch, "compile")
        ):
            self._compiled_forward_impl = torch.compile(self._forward_impl)

        self.post_init()

    @property
    def target_layer_ids(self) -> list[int]:
        if self.config.eagle_aux_hidden_state_layer_ids is None:
            raise ValueError("eagle1_train target_layer_ids have not been resolved.")
        return self.config.eagle_aux_hidden_state_layer_ids

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        model = super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        verifier_path = model.config.speculators_config.verifier.name_or_path
        if verifier_path is not None:
            model.config.eagle_aux_hidden_state_layer_ids = (
                _resolve_eagle1_target_layer_ids(
                    model.config.eagle_aux_hidden_state_layer_ids,
                    verifier_path,
                )
            )
        return model

    def load_verifier_weights(self) -> None:
        super().load_verifier_weights()

        verifier_config = self.config.speculators_config.verifier
        if verifier_config.name_or_path is None:
            return

        verifier_model_config = get_verifier_config(verifier_config.name_or_path)
        if verifier_model_config.hidden_size != self.hidden_size:
            raise ValueError(
                f"Verifier hidden size {verifier_model_config.hidden_size} does not "
                f"match draft hidden size {self.hidden_size}."
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        lengths: torch.Tensor | None = None,
        loss_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        verifier_last_hidden_states: torch.Tensor | None = None,
        ttt_steps: int = 3,
        ttt_step_loss_decay: float = 1.0,
        use_off_policy_tokens: bool = False,
        loss_fn=kl_div_loss,
        **kwargs: Any,
    ):
        impl = self._compiled_forward_impl or self._forward_impl
        return impl(
            hidden_states=hidden_states,
            input_ids=input_ids,
            lengths=lengths,
            loss_mask=loss_mask,
            position_ids=position_ids,
            verifier_last_hidden_states=verifier_last_hidden_states,
            ttt_steps=ttt_steps,
            ttt_step_loss_decay=ttt_step_loss_decay,
            use_off_policy_tokens=use_off_policy_tokens,
            loss_fn=loss_fn,
            **kwargs,
        )

    def _forward_impl(  # noqa: C901
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        lengths: torch.Tensor | None = None,
        loss_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        verifier_last_hidden_states: torch.Tensor | None = None,
        ttt_steps: int = 3,
        ttt_step_loss_decay: float = 1.0,
        use_off_policy_tokens: bool = False,
        loss_fn=kl_div_loss,
        **kwargs: Any,
    ):
        del kwargs
        loss_fn = loss_fn or kl_div_loss
        device = hidden_states.device
        total_seq_len = hidden_states.shape[1]

        if hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(
                "eagle1_train expects exactly one aux hidden-state layer. "
                f"Got hidden_states.shape[-1]={hidden_states.shape[-1]} and "
                f"hidden_size={self.hidden_size}."
            )

        input_ids = input_ids.long()
        if lengths is None:
            lengths = torch.tensor([total_seq_len], dtype=torch.long, device=device)
        if position_ids is None:
            position_ids = torch.arange(
                total_seq_len, dtype=torch.long, device=device
            ).unsqueeze(0)

        attention_mask = _packed_causal_mask(
            lengths=lengths,
            seq_len=total_seq_len,
            dtype=hidden_states.dtype,
            device=device,
        )

        original_input_ids = input_ids.detach().clone()
        return_loss = verifier_last_hidden_states is not None
        if return_loss:
            with torch.no_grad():
                targets = self.verifier_lm_head(
                    self.verifier_norm(verifier_last_hidden_states)
                )

            loss = torch.tensor(0.0, device=device)
            prev_correct = (
                loss_mask.clone()
                if loss_mask is not None
                else torch.ones(1, total_seq_len, device=device, dtype=torch.bool)
            )
            metrics = {}

        draft_tokens = []
        current_hidden = hidden_states
        for ttt_step in range(ttt_steps):
            with torch.no_grad():
                input_embeds = self.embed_tokens(input_ids)

            current_hidden = self.fusion_fc(
                torch.cat([input_embeds, current_hidden], dim=-1)
            )
            position_embeddings = self.rotary_emb(current_hidden, position_ids)

            for decoder_layer in self.layers:
                current_hidden = decoder_layer(
                    hidden_states=current_hidden,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                )

            logits = self.lm_head(self.norm(current_hidden))

            if return_loss:
                s_loss, s_metrics = compute_metrics(
                    logits,
                    targets,
                    loss_mask,
                    prev_correct,
                    ttt_step,
                    ttt_step_loss_decay,
                    loss_fn=loss_fn,
                )
                loss = loss + s_loss
                metrics.update(s_metrics)

            input_ids = torch.argmax(logits, dim=-1)
            draft_tokens.append(input_ids.detach().clone())

            if self.d2t is not None:
                input_ids = input_ids + self.d2t[input_ids]  # type: ignore[index]

            if use_off_policy_tokens:
                input_ids = torch.cat(
                    [
                        original_input_ids[:, 1 + ttt_step :],
                        original_input_ids.new_zeros(1, 1 + ttt_step),
                    ],
                    dim=-1,
                )

        if return_loss:
            metrics["loss_sum"] = loss.detach().clone()
            metrics["loss_total"] = torch.tensor(1.0, device=device)
            return draft_tokens, loss, metrics

        return draft_tokens

    @classmethod
    def from_training_args(
        cls,
        verifier_config: PretrainedConfig,
        t2d: torch.Tensor | None = None,
        d2t: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> "Eagle1TrainDraftModel":
        target_layer_ids = _resolve_eagle1_target_layer_ids(
            kwargs.get("target_layer_ids"), kwargs["verifier_name_or_path"]
        )

        config = Eagle1TrainSpeculatorConfig(
            transformer_layer_config=verifier_config,
            draft_vocab_size=kwargs["draft_vocab_size"],
            eagle_aux_hidden_state_layer_ids=target_layer_ids,
            speculators_config=SpeculatorsConfig(
                algorithm="eagle",
                proposal_methods=[
                    GreedyTokenProposalConfig(
                        speculative_tokens=kwargs["ttt_steps"],
                    )
                ],
                default_proposal_method="greedy",
                verifier=VerifierConfig.from_config(
                    verifier_config,
                    name_or_path=kwargs["verifier_name_or_path"],
                ),
            ),
        )
        model = cls(config=config)
        model.load_vocab_mappings(t2d, d2t)
        model.load_verifier_weights()
        return model

    @staticmethod
    def get_trainer_kwargs(**kwargs) -> tuple[dict, dict]:
        loss_fn = resolve_loss_fn(kwargs["loss_fn"])
        train_kwargs = {
            "use_off_policy_tokens": kwargs["use_off_policy_tokens"],
            "ttt_steps": kwargs["ttt_steps"],
            "ttt_step_loss_decay": kwargs["ttt_step_loss_decay"],
            "loss_fn": loss_fn,
        }
        val_kwargs = {
            "use_off_policy_tokens": False,
            "ttt_steps": kwargs["ttt_steps"],
            "ttt_step_loss_decay": kwargs["ttt_step_loss_decay"],
            "loss_fn": loss_fn,
        }
        return train_kwargs, val_kwargs
