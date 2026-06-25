"""Unit tests for the EAGLE1 training-only draft model."""

import torch
from transformers.models.llama.configuration_llama import LlamaConfig

from speculators import SpeculatorsConfig, VerifierConfig
from speculators.models.eagle1_train import (
    Eagle1TrainDraftModel,
    Eagle1TrainSpeculatorConfig,
)
from speculators.proposals import GreedyTokenProposalConfig


def _make_eagle1_model() -> Eagle1TrainDraftModel:
    config = Eagle1TrainSpeculatorConfig(
        transformer_layer_config=LlamaConfig(
            attention_bias=False,
            attention_dropout=0.0,
            bos_token_id=1,
            eos_token_id=2,
            head_dim=128,
            hidden_act="silu",
            hidden_size=2048,
            initializer_range=0.02,
            intermediate_size=6144,
            max_position_embeddings=1024,
            mlp_bias=False,
            num_attention_heads=32,
            num_hidden_layers=1,
            num_key_value_heads=4,
            rms_norm_eps=1e-6,
            tie_word_embeddings=False,
            vocab_size=128,
        ),
        draft_vocab_size=128,
        eagle_aux_hidden_state_layer_ids=[47],
        speculators_config=SpeculatorsConfig(
            algorithm="eagle",
            proposal_methods=[GreedyTokenProposalConfig(speculative_tokens=4)],
            default_proposal_method="greedy",
            verifier=VerifierConfig(
                name_or_path=None,
                architectures=["Qwen3MoeForCausalLM"],
            ),
        ),
    )
    model = Eagle1TrainDraftModel(config)
    torch.nn.init.normal_(model.embed_tokens.weight, std=0.02)
    torch.nn.init.normal_(model.lm_head.weight, std=0.02)
    torch.nn.init.normal_(model.verifier_lm_head.weight, std=0.02)
    model.train()
    return model


def test_eagle1_hass_loss_forward_backward_has_feature_and_token_metrics(seed):
    model = _make_eagle1_model()
    input_ids = torch.randint(0, model.verifier_vocab_size, (1, 8))
    hidden_states = torch.randn(1, 8, model.hidden_size)
    verifier_last_hidden_states = torch.randn(1, 8, model.hidden_size)
    loss_mask = torch.ones(1, 8, dtype=torch.bool)

    draft_tokens, loss, metrics = model(
        hidden_states=hidden_states,
        input_ids=input_ids,
        loss_mask=loss_mask,
        verifier_last_hidden_states=verifier_last_hidden_states,
        ttt_steps=4,
        loss_mode="eagle1_hass",
    )

    assert len(draft_tokens) == 4
    assert torch.isfinite(loss)
    assert loss.requires_grad
    assert {"vloss_0_sum", "ploss_0_sum", "loss_0_sum"} <= set(metrics)

    loss.backward()
    assert model.fusion_fc.weight.grad is not None
    assert torch.isfinite(model.fusion_fc.weight.grad).all()


def test_eagle1_train_matches_vllm_legacy_eagle_shape():
    model = _make_eagle1_model()

    assert not hasattr(model, "norm")
    assert isinstance(model.layers[0].input_layernorm, torch.nn.Identity)
