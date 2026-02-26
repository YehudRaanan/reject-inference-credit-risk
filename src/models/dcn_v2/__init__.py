# DCN-v2 (Deep & Cross Network v2) for Credit Risk Assessment
#
# This module implements DCN-v2 architecture to address the information bottleneck
# identified in VIME (Phase 5.7). Key features:
# - No dimensional compression (preserves all 253 features via residual connections)
# - Explicit polynomial feature interactions (Cross Network)
# - Native missingness handling (dedicated Missing Bin)
#
# Architecture:
#   Input -> SoftBinning -> EmbeddingEngine -> x₀
#   x₀ -> CrossNetwork (explicit interactions) + DeepNetwork (implicit) -> concat -> head
#
# Reference: Wang et al., "DCN V2: Improved Deep & Cross Network" (2020)
#
# Created: 2026-02-15
# Status: Phase 8 implementation (pair-programming with owner)

from .soft_binning import SoftBinning, compute_quantiles_for_binning
from .embedding_engine import EmbeddingEngine
from .cross_network import CrossLayer, CrossNetwork
from .deep_network import DeepNetwork
from .dcn_v2 import DCNv2
from .data_utils import (
    DCNv2Dataset,
    dcnv2_collate_fn,
    load_dcn_data,
    load_cardinalities,
    create_dcn_dataloaders,
)
from .trainer import (
    FocalLoss,
    precision_at_k,
    evaluate_full,
    evaluate,
    train_epoch,
    train_dcn_model,
)

__all__ = [
    # Model components
    'SoftBinning',
    'compute_quantiles_for_binning',
    'EmbeddingEngine',
    'CrossLayer',
    'CrossNetwork',
    'DeepNetwork',
    'DCNv2',
    # Data utilities
    'DCNv2Dataset',
    'dcnv2_collate_fn',
    'load_dcn_data',
    'load_cardinalities',
    'create_dcn_dataloaders',
    # Training utilities
    'FocalLoss',
    'precision_at_k',
    'evaluate_full',
    'evaluate',
    'train_epoch',
    'train_dcn_model',
]
