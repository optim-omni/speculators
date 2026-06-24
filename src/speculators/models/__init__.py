from speculators.models.attention import ALL_ATTENTION_FUNCTIONS  # noqa: F401

from .dflash import DFlashDraftModel, DFlashSpeculatorConfig
from .eagle1_train import Eagle1TrainDraftModel, Eagle1TrainSpeculatorConfig
from .eagle3 import Eagle3DraftModel, Eagle3SpeculatorConfig
from .mtp import MTPDraftModel, MTPSpeculatorConfig
from .peagle import PEagleDraftModel, PEagleSpeculatorConfig

__all__ = [
    "DFlashDraftModel",
    "DFlashSpeculatorConfig",
    "Eagle1TrainDraftModel",
    "Eagle1TrainSpeculatorConfig",
    "Eagle3DraftModel",
    "Eagle3SpeculatorConfig",
    "MTPDraftModel",
    "MTPSpeculatorConfig",
    "PEagleDraftModel",
    "PEagleSpeculatorConfig",
]
