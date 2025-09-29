# -*- coding: utf-8 -*-
"""
DCoTAgentAligner: Diverse Chain-of-Thought Agent Alignment Module

This package provides tools for generating, clustering, and evaluating
diverse reasoning chains with semantic graph alignment capabilities.
"""

from .DCoTAgentAligner import DCoTAgentAligner
from .embed_chain import embed_chain
from .extract_graph_nodes import extract_graph_nodes
from .compute_diversity import compute_diversity
from .validate_dcot_submission import validate_dcot_submission

__version__ = "1.0.0"
__author__ = "AI Research Agent Team"

__all__ = [
    'DCoTAgentAligner',
    'embed_chain',
    'extract_graph_nodes',
    'compute_diversity',
    'validate_dcot_submission'
]
