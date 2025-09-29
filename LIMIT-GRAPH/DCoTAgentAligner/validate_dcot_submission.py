# -*- coding: utf-8 -*-
"""
Validation module for DCoT submissions
"""

import json
import os
from typing import List, Dict, Any
try:
    from .compute_diversity import compute_diversity
    from .extract_graph_nodes import extract_graph_nodes
except ImportError:
    # Fallback for direct execution
    from compute_diversity import compute_diversity
    from extract_graph_nodes import extract_graph_nodes

def load_chains(submission_path: str) -> List[Dict[str, Any]]:
    """Load reasoning chains from submission file"""
    if not os.path.exists(submission_path):
        raise FileNotFoundError(f"Submission file not found: {submission_path}")
    
    with open(submission_path, 'r', encoding='utf-8') as f:
        if submission_path.endswith('.json'):
            data = json.load(f)
            return data.get('chains', [])
        else:
            # Assume text file with one chain per line
            lines = f.readlines()
            return [{'text': line.strip(), 'id': i} for i, line in enumerate(lines)]

def evaluate_graph_alignment(chains: List[Dict[str, Any]], lang: str = 'id') -> float:
    """Evaluate how well chains align with graph structures"""
    if not chains:
        return 0.0
    
    # Sample graph vocabulary for evaluation
    graph_vocab = {
        'id': ['analisis', 'logika', 'sistem', 'pola', 'bukti', 'hipotesis'],
        'en': ['analysis', 'logic', 'system', 'pattern', 'evidence', 'hypothesis'],
        'zh': ['分析', '逻辑', '系统', '模式', '证据', '假设']
    }
    
    vocab = graph_vocab.get(lang, graph_vocab['en'])
    total_alignment = 0
    
    for chain in chains:
        text = chain.get('text', '')
        nodes = extract_graph_nodes(text, vocab)
        alignment = len(nodes) / len(vocab) if vocab else 0
        total_alignment += alignment
    
    return total_alignment / len(chains)

def evaluate_lang_coverage(chains: List[Dict[str, Any]], target_lang: str = 'id') -> float:
    """Evaluate multilingual coverage and accuracy"""
    if not chains:
        return 0.0
    
    # Language-specific keywords for validation
    lang_keywords = {
        'id': ['adalah', 'dengan', 'untuk', 'dari', 'yang', 'ini', 'itu'],
        'en': ['is', 'with', 'for', 'from', 'that', 'this', 'the'],
        'zh': ['是', '与', '为', '从', '那', '这', '的']
    }
    
    keywords = lang_keywords.get(target_lang, lang_keywords['en'])
    correct_lang_chains = 0
    
    for chain in chains:
        text = chain.get('text', '').lower()
        keyword_matches = sum(1 for keyword in keywords if keyword in text)
        
        # Consider chain as correct language if it has at least 2 keyword matches
        if keyword_matches >= 2:
            correct_lang_chains += 1
    
    return correct_lang_chains / len(chains)

def validate_dcot_submission(submission_path: str, lang: str = 'id') -> Dict[str, float]:
    """
    Validate a DCoT submission file
    
    Args:
        submission_path: Path to submission file
        lang: Target language for validation
        
    Returns:
        Dictionary with validation scores
    """
    try:
        chains = load_chains(submission_path)
        
        if not chains:
            return {
                'diversity': 0.0,
                'graph_consistency': 0.0,
                'multilingual_accuracy': 0.0,
                'overall_score': 0.0
            }
        
        # Extract text for diversity computation
        texts = [chain.get('text', '') for chain in chains if chain.get('text')]
        
        diversity_score = compute_diversity(texts) if texts else 0.0
        graph_score = evaluate_graph_alignment(chains, lang)
        multilingual_score = evaluate_lang_coverage(chains, lang)
        
        # Calculate overall score
        overall_score = (diversity_score + graph_score + multilingual_score) / 3
        
        return {
            'diversity': diversity_score,
            'graph_consistency': graph_score,
            'multilingual_accuracy': multilingual_score,
            'overall_score': overall_score
        }
        
    except Exception as e:
        print(f"Validation error: {e}")
        return {
            'diversity': 0.0,
            'graph_consistency': 0.0,
            'multilingual_accuracy': 0.0,
            'overall_score': 0.0,
            'error': str(e)
        }
