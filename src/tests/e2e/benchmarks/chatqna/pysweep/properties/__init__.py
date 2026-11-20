# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Concrete configuration properties (one knob per class)."""

from .llm_model import LlmModel
from .max_tokens import MaxNewTokens
from .reranker_top_n import RerankerTopN
from .retriever_k import RetrieverK

__all__ = ["LlmModel", "MaxNewTokens", "RerankerTopN", "RetrieverK"]
