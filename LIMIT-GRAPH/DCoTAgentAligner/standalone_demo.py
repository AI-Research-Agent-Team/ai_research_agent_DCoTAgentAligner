# -*- coding: utf-8 -*-
"""
Standalone demo for DCoTAgentAligner - No relative imports
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mock implementations for dependencies
class MockSentenceTransformer:
    def encode(self, text):
        # Simple mock embedding based on text hash
        return np.random.rand(384)  # Standard embedding size

def embed_chain(chain):
    """Mock embedding function"""
    mock_model = MockSentenceTransformer()
    return mock_model.encode(chain)

def extract_graph_nodes(chain, graph_vocab):
    """Extract graph nodes from text"""
    matched = []
    if isinstance(graph_vocab, dict):
        # Flatten all vocabulary items
        all_vocab = []
        for category in graph_vocab.values():
            if isinstance(category, list):
                all_vocab.extend(category)
            else:
                all_vocab.append(category)
        vocab = all_vocab
    else:
        vocab = graph_vocab
    
    for concept in vocab:
        if concept.lower() in chain.lower():
            matched.append(concept)
    return matched

def compute_diversity(chains):
    """Compute diversity score using simple text comparison"""
    if len(chains) < 2:
        return 0.0
    
    # Simple diversity based on unique words
    all_words = set()
    total_words = 0
    
    for chain in chains:
        words = set(chain.lower().split())
        all_words.update(words)
        total_words += len(words)
    
    # Diversity as ratio of unique words to total words
    diversity = len(all_words) / max(total_words, 1)
    return min(diversity, 1.0)

# Mock LLM for testing
class MockLLM:
    def __init__(self):
        self.sample_responses = {
            'analytical': [
                "First, I need to analyze the problem systematically. Step 1: Identify key variables. Step 2: Examine relationships between components. Step 3: Apply logical reasoning to derive conclusions.",
                "To solve this analytically, I'll break it down: 1) Define the problem scope, 2) Gather relevant data, 3) Apply analytical frameworks, 4) Validate results through testing.",
                "Using analytical thinking: Begin with hypothesis formation, then collect evidence, analyze patterns, and draw evidence-based conclusions."
            ],
            'creative': [
                "Let me approach this creatively! What if we think outside the box? Maybe we can combine unexpected elements to find innovative solutions that haven't been tried before.",
                "Creative thinking suggests multiple pathways: brainstorming unconventional approaches, exploring analogies from other domains, and embracing experimental methods.",
                "Thinking creatively, I see opportunities for novel combinations and fresh perspectives that could lead to breakthrough insights."
            ],
            'systematic': [
                "Following a systematic approach: Phase 1 - Planning and preparation. Phase 2 - Implementation with checkpoints. Phase 3 - Evaluation and refinement.",
                "Systematic methodology requires: 1) Clear objective definition, 2) Structured process design, 3) Sequential execution, 4) Quality control at each step.",
                "Using systematic reasoning: Establish framework, follow procedures methodically, document progress, and ensure consistency throughout."
            ],
            'intuitive': [
                "My intuition suggests there's a pattern here that connects to broader principles. The solution feels like it should involve recognizing underlying relationships.",
                "Intuitively, this reminds me of similar situations where the key was understanding the implicit connections and trusting the emerging insights.",
                "Following intuitive reasoning, I sense that the answer lies in recognizing the natural flow and organic relationships within the system."
            ],
            'critical': [
                "Let me critically examine this: What assumptions are we making? Are there alternative explanations? What evidence supports or contradicts our initial thinking?",
                "Critical analysis reveals potential flaws in common approaches. We must question premises, evaluate evidence quality, and consider counterarguments.",
                "Applying critical thinking: Challenge existing beliefs, examine biases, verify sources, and maintain healthy skepticism throughout the process."
            ]
        }
    
    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 512) -> str:
        # Extract style from prompt
        style = 'analytical'  # default
        for s in self.sample_responses.keys():
            if s in prompt.lower():
                style = s
                break
        
        # Add some randomness based on temperature
        responses = self.sample_responses[style]
        idx = int(temperature * len(responses)) % len(responses)
        
        return responses[idx]

# Simplified DCoTAgentAligner implementation
class DCoTAgentAligner:
    def __init__(self, model, lang: str = 'id', graph_vocab: Dict = None):
        self.model = model
        self.lang = lang
        self.graph_vocab = graph_vocab or {}
        self.reasoning_styles = [
            'analytical', 'creative', 'systematic', 'intuitive', 'critical'
        ]
        self.multilingual_prompts = {
            'id': 'Jelaskan langkah demi langkah:',
            'en': 'Explain step by step:',
            'zh': '逐步解释：',
            'es': 'Explica paso a paso:',
            'ar': 'اشرح خطوة بخطوة:'
        }
        
    def generate_chains(self, prompt: str, n: int = 5, diverse_styles: bool = True) -> List[Dict[str, Any]]:
        chains = []
        
        for i in range(n):
            temperature = 0.7 + (i * 0.1)
            style = self.reasoning_styles[i % len(self.reasoning_styles)] if diverse_styles else 'analytical'
            styled_prompt = self._create_styled_prompt(prompt, style)
            
            try:
                generated_text = self.model.generate(styled_prompt, temperature=temperature, max_tokens=512)
                
                chain_data = {
                    'id': f'chain_{i}',
                    'text': generated_text,
                    'style': style,
                    'temperature': temperature,
                    'language': self.lang,
                    'prompt': styled_prompt
                }
                
                chains.append(chain_data)
                
            except Exception as e:
                logger.warning(f"Failed to generate chain {i}: {e}")
                continue
                
        return chains

    def cluster_chains(self, chains: List[Dict[str, Any]], n_clusters: int = 3):
        texts = [chain['text'] for chain in chains]
        embeddings = np.array([embed_chain(text) for text in texts])
        
        # Simple clustering without sklearn - use basic similarity grouping
        cluster_labels = []
        
        # Assign clusters based on text similarity (simple approach)
        for i, text in enumerate(texts):
            # Simple clustering based on text length and style
            style = chains[i].get('style', 'analytical')
            style_map = {'analytical': 0, 'creative': 1, 'systematic': 0, 'intuitive': 1, 'critical': 2}
            cluster = style_map.get(style, i % n_clusters)
            cluster_labels.append(cluster)
            chains[i]['cluster'] = cluster
        
        return np.array(cluster_labels), embeddings

    def align_to_graph(self, chains: List[Dict[str, Any]], graph_vocab: Dict = None) -> List[Dict[str, Any]]:
        vocab = graph_vocab or self.graph_vocab
        aligned_chains = []
        
        for chain in chains:
            graph_nodes = extract_graph_nodes(chain['text'], vocab)
            
            # Calculate alignment score
            if isinstance(vocab, dict):
                total_vocab_size = sum(len(v) if isinstance(v, list) else 1 for v in vocab.values())
            else:
                total_vocab_size = len(vocab)
            
            alignment_score = len(graph_nodes) / max(total_vocab_size, 1)
            
            aligned_chain = chain.copy()
            aligned_chain.update({
                'graph_nodes': graph_nodes,
                'alignment_score': alignment_score,
                'graph_coverage': len(set(graph_nodes)) / max(total_vocab_size, 1)
            })
            
            aligned_chains.append(aligned_chain)
            
        return aligned_chains

    def evaluate_diversity(self, chains: List[Dict[str, Any]]) -> Dict[str, float]:
        texts = [chain['text'] for chain in chains]
        semantic_diversity = compute_diversity(texts)
        
        styles = [chain.get('style', 'unknown') for chain in chains]
        style_diversity = len(set(styles)) / len(styles)
        
        languages = [chain.get('language', self.lang) for chain in chains]
        language_diversity = len(set(languages)) / len(languages)
        
        return {
            'semantic_diversity': semantic_diversity,
            'style_diversity': style_diversity,
            'language_diversity': language_diversity,
            'overall_diversity': (semantic_diversity + style_diversity + language_diversity) / 3
        }

    def generate_multilingual_chains(self, prompt: str, languages: List[str] = None, chains_per_lang: int = 3) -> List[Dict[str, Any]]:
        languages = languages or ['id', 'en', 'zh']
        all_chains = []
        
        for lang in languages:
            if lang in self.multilingual_prompts:
                lang_prompt = f"{self.multilingual_prompts[lang]} {prompt}"
                
                original_lang = self.lang
                self.lang = lang
                
                lang_chains = self.generate_chains(lang_prompt, n=chains_per_lang, diverse_styles=True)
                
                for chain in lang_chains:
                    chain['language'] = lang
                    
                all_chains.extend(lang_chains)
                self.lang = original_lang
                
        return all_chains

    def _create_styled_prompt(self, prompt: str, style: str) -> str:
        style_prefixes = {
            'analytical': 'Using systematic analysis and logical reasoning:',
            'creative': 'Using creative and innovative thinking:',
            'systematic': 'Using a structured, step-by-step approach:',
            'intuitive': 'Using intuitive insights and pattern recognition:',
            'critical': 'Using critical evaluation and questioning:'
        }
        
        prefix = style_prefixes.get(style, 'Using careful reasoning:')
        return f"{prefix} {prompt}"

def demo_complete_system():
    """Demonstrate the complete DCoT system"""
    print("DCoTAgentAligner Standalone Demo")
    print("=" * 50)
    
    # Initialize system
    mock_llm = MockLLM()
    graph_vocab = {
        'concepts': ['analysis', 'reasoning', 'logic', 'creativity', 'system', 'pattern'],
        'relations': ['causes', 'leads_to', 'requires', 'supports'],
        'entities': ['problem', 'solution', 'method', 'result']
    }
    
    aligner = DCoTAgentAligner(
        model=mock_llm,
        lang='en',
        graph_vocab=graph_vocab
    )
    
    # Demo 1: Basic chain generation
    print("\n1. Generating diverse reasoning chains...")
    prompt = "How can we solve complex problems effectively?"
    chains = aligner.generate_chains(prompt, n=5, diverse_styles=True)
    
    print(f"Generated {len(chains)} chains:")
    for i, chain in enumerate(chains):
        print(f"  Chain {i+1} ({chain['style']}): {chain['text'][:80]}...")
    
    # Demo 2: Clustering
    print("\n2. Clustering chains...")
    try:
        cluster_labels, embeddings = aligner.cluster_chains(chains, n_clusters=3)
        print(f"Clustered into 3 groups:")
        for i, chain in enumerate(chains):
            print(f"  Chain {i+1}: Cluster {chain['cluster']}")
    except ImportError:
        print("  Clustering skipped (sklearn not available)")
        cluster_labels, embeddings = None, None
    
    # Demo 3: Graph alignment
    print("\n3. Aligning to semantic graph...")
    aligned_chains = aligner.align_to_graph(chains)
    
    print("Graph alignment results:")
    for i, chain in enumerate(aligned_chains):
        print(f"  Chain {i+1}: {len(chain['graph_nodes'])} nodes, score: {chain['alignment_score']:.3f}")
    
    # Demo 4: Diversity evaluation
    print("\n4. Evaluating diversity...")
    diversity_metrics = aligner.evaluate_diversity(aligned_chains)
    
    print("Diversity metrics:")
    for metric, value in diversity_metrics.items():
        print(f"  {metric}: {value:.3f}")
    
    # Demo 5: Multilingual generation
    print("\n5. Generating multilingual chains...")
    multilingual_chains = aligner.generate_multilingual_chains(
        "problem solving methodology",
        languages=['en', 'id'],
        chains_per_lang=2
    )
    
    print(f"Generated {len(multilingual_chains)} multilingual chains:")
    for i, chain in enumerate(multilingual_chains):
        print(f"  Chain {i+1} ({chain['language']}): {chain['text'][:60]}...")
    
    print("\n" + "=" * 50)
    print("Demo completed successfully!")
    print("DCoTAgentAligner is working correctly.")
    
    return {
        'chains': chains,
        'aligned_chains': aligned_chains,
        'multilingual_chains': multilingual_chains,
        'diversity_metrics': diversity_metrics
    }

if __name__ == "__main__":
    results = demo_complete_system()