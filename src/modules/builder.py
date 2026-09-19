# Copyright 2026 Muhammad Nizwa
# SPDX-License-Identifier: Apache-2.0

import torch

from transformers import PreTrainedTokenizerBase

from src.modules.two_tower import (
    NewsEncoder,
    UserEncoder,
    TwoTowerModel,
)
from config import TwoTowerConfig



def build_two_tower_model(
    model_config: TwoTowerConfig,
    category_vocab_size: int,
    subcategory_vocab_size: int,
    text_embedding_weights: torch.Tensor | None,
    text_embedding_dim: int,
    tokenizer: PreTrainedTokenizerBase,
    entity_embedding_weights: torch.Tensor | None,
    entity_embedding_dim: int | None,
):
    news_tower_config = model_config.news_tower
    user_tower_config = model_config.user_tower

    news_encoder = NewsEncoder(
        vocab_size=tokenizer.vocab_size,
        d_model=news_tower_config.d_model,
        num_heads=news_tower_config.num_heads,
        num_layers=news_tower_config.num_layers,
        d_ff=news_tower_config.d_ff,
        pool_hidden_dim=news_tower_config.pool_hidden_dim,
        text_embedding_dim=text_embedding_dim,
        text_embedding_weights=text_embedding_weights,
        entity_embedding_dim=entity_embedding_dim,
        entity_embedding_weights=entity_embedding_weights,
        category_vocab_size=category_vocab_size,
        category_embedding_dim=news_tower_config.category_embedding_dim,
        subcategory_vocab_size=subcategory_vocab_size,
        subcategory_embedding_dim=news_tower_config.subcategory_embedding_dim,
        dropout=news_tower_config.dropout,
    )

    user_encoder = UserEncoder(
        input_dim=news_tower_config.d_model,
        d_model=user_tower_config.d_model,
        num_layers=user_tower_config.num_layers,
        output_dim=news_tower_config.d_model,
        dropout=user_tower_config.dropout,
        bidirectional=user_tower_config.bidirectional,
    )

    model = TwoTowerModel(
        news_tower=news_encoder,
        user_tower=user_encoder,
        normalize_embeddings=model_config.normalize_embeddings,
    )

    return model
