# -*- coding: utf-8 -*-
"""
Demo script for DCoTAgentAligner - Diverse Chain-of-Thought Agent Alignment

This script demonstrates the complete functionality of the DCoTAgentAligner
including chain generation, clustering, graph alignment, and multilingual support.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MockLLM:
    """Mock LLM for demonstration purposes"""
    
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
        """Generate mock response based on prompt style"""
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

def create_sample_graph_vocab() -> Dict[str, List[str]]:
    """Create sample graph vocabulary for demonstration"""
    return {
        'concepts': ['analysis', 'reasoning', 'logic', 'creativity', 'system', 'pattern', 'evidence', 'hypothesis'],
        'relations': ['causes', 'leads_to', 'requires', 'supports', 'contradicts', 'enhances'],
        'entities': ['problem', 'solution', 'method', 'result', 'process', 'framework']
    }

def demo_basic_chain_generation():
    """Demonstrate basic chain generation functionality"""
    print("\n=== Demo: Basic Chain Generation ===")
    
    from DCoTAgentAligner import DCoTAgentAligner
    
    # Initialize with mock model
    mock_llm = MockLLM()
    graph_vocab = create_sample_graph_vocab()
    
    aligner = DCoTAgentAligner(
        model=mock_llm,
        lang='en',
        graph_vocab=graph_vocab
    )
    
    # Generate diverse reasoning chains
    prompt = "How can we solve complex problems effectively?"
    chains = aligner.generate_chains(prompt, n=5, diverse_styles=True)
    
    print(f"Generated {len(chains)} reasoning chains:")
    for i, chain in enumerate(chains):
        print(f"\nChain {i+1} ({chain['style']}):")
        print(f"Text: {chain['text'][:100]}...")
        print(f"Temperature: {chain['temperature']}")
    
    return chains

def demo_clustering_and_visualization():
    """Demonstrate clustering and visualization"""
    print("\n=== Demo: Clustering and Visualization ===")
    
    from DCoTAgentAligner import DCoTAgentAligner
    
    mock_llm = MockLLM()
    graph_vocab = create_sample_graph_vocab()
    
    aligner = DCoTAgentAligner(
        model=mock_llm,
        lang='en',
        graph_vocab=graph_vocab
    )
    
    # Generate chains
    prompt = "Explain the process of scientific discovery"
    chains = aligner.generate_chains(prompt, n=8, diverse_styles=True)
    
    # Cluster chains
    cluster_labels, embeddings = aligner.cluster_chains(chains, n_clusters=3)
    
    print(f"Clustered {len(chains)} chains into 3 groups:")
    for i, chain in enumerate(chains):
        print(f"Chain {i+1}: Cluster {chain['cluster']}, Style: {chain['style']}")
    
    # Visualize clusters
    try:
        aligner.visualize_clusters(chains, embeddings, save_path='dcot_clusters.png')
        print("Visualization saved as 'dcot_clusters.png'")
    except Exception as e:
        print(f"Visualization skipped: {e}")
    
    return chains, embeddings

def demo_graph_alignment():
    """Demonstrate graph alignment functionality"""
    print("\n=== Demo: Graph Alignment ===")
    
    from DCoTAgentAligner import DCoTAgentAligner
    
    mock_llm = MockLLM()
    graph_vocab = create_sample_graph_vocab()
    
    aligner = DCoTAgentAligner(
        model=mock_llm,
        lang='en',
        graph_vocab=graph_vocab
    )
    
    # Generate chains
    prompt = "Describe logical reasoning and systematic analysis methods"
    chains = aligner.generate_chains(prompt, n=5, diverse_styles=True)
    
    # Align to graph
    aligned_chains = aligner.align_to_graph(chains, graph_vocab)
    
    print("Graph alignment results:")
    for i, chain in enumerate(aligned_chains):
        print(f"\nChain {i+1}:")
        print(f"Style: {chain['style']}")
        print(f"Graph nodes found: {chain['graph_nodes']}")
        print(f"Alignment score: {chain['alignment_score']:.3f}")
        print(f"Graph coverage: {chain['graph_coverage']:.3f}")
    
    return aligned_chains

def demo_diversity_evaluation():
    """Demonstrate diversity evaluation"""
    print("\n=== Demo: Diversity Evaluation ===")
    
    from DCoTAgentAligner import DCoTAgentAligner
    
    mock_llm = MockLLM()
    graph_vocab = create_sample_graph_vocab()
    
    aligner = DCoTAgentAligner(
        model=mock_llm,
        lang='en',
        graph_vocab=graph_vocab
    )
    
    # Generate diverse chains
    prompt = "What are effective problem-solving strategies?"
    chains = aligner.generate_chains(prompt, n=6, diverse_styles=True)
    
    # Evaluate diversity
    diversity_metrics = aligner.evaluate_diversity(chains)
    
    print("Diversity evaluation results:")
    for metric, value in diversity_metrics.items():
        print(f"{metric}: {value:.3f}")
    
    return diversity_metrics

def demo_multilingual_chains():
    """Demonstrate multilingual chain generation"""
    print("\n=== Demo: Multilingual Chain Generation ===")
    
    from DCoTAgentAligner import DCoTAgentAligner
    
    mock_llm = MockLLM()
    graph_vocab = create_sample_graph_vocab()
    
    aligner = DCoTAgentAligner(
        model=mock_llm,
        lang='en',
        graph_vocab=graph_vocab
    )
    
    # Generate multilingual chains
    prompt = "problem solving methodology"
    languages = ['en', 'id', 'zh']
    multilingual_chains = aligner.generate_multilingual_chains(
        prompt, 
        languages=languages,
        chains_per_lang=2
    )
    
    print(f"Generated {len(multilingual_chains)} multilingual chains:")
    for i, chain in enumerate(multilingual_chains):
        print(f"\nChain {i+1}:")
        print(f"Language: {chain['language']}")
        print(f"Style: {chain['style']}")
        print(f"Prompt: {chain['prompt'][:50]}...")
    
    # Evaluate multilingual diversity
    diversity_metrics = aligner.evaluate_diversity(multilingual_chains)
    print(f"\nMultilingual diversity metrics:")
    for metric, value in diversity_metrics.items():
        print(f"{metric}: {value:.3f}")
    
    return multilingual_chains

def demo_complete_pipeline():
    """Demonstrate complete DCoT pipeline"""
    print("\n=== Demo: Complete DCoT Pipeline ===")
    
    from DCoTAgentAligner import DCoTAgentAligner
    
    mock_llm = MockLLM()
    graph_vocab = create_sample_graph_vocab()
    
    aligner = DCoTAgentAligner(
        model=mock_llm,
        lang='en',
        graph_vocab=graph_vocab
    )
    
    # Step 1: Generate diverse chains
    prompt = "How do we approach complex research questions?"
    chains = aligner.generate_chains(prompt, n=10, diverse_styles=True)
    print(f"Step 1: Generated {len(chains)} diverse reasoning chains")
    
    # Step 2: Cluster chains
    cluster_labels, embeddings = aligner.cluster_chains(chains, n_clusters=4)
    print(f"Step 2: Clustered chains into 4 groups")
    
    # Step 3: Align to graph
    aligned_chains = aligner.align_to_graph(chains, graph_vocab)
    print(f"Step 3: Aligned chains to semantic graph")
    
    # Step 4: Evaluate diversity
    diversity_metrics = aligner.evaluate_diversity(aligned_chains)
    print(f"Step 4: Evaluated diversity - Overall: {diversity_metrics['overall_diversity']:.3f}")
    
    # Step 5: Generate multilingual variants
    multilingual_chains = aligner.generate_multilingual_chains(
        prompt, 
        languages=['en', 'id'],
        chains_per_lang=3
    )
    print(f"Step 5: Generated {len(multilingual_chains)} multilingual chains")
    
    # Final summary
    print(f"\nPipeline Summary:")
    print(f"- Total chains generated: {len(chains) + len(multilingual_chains)}")
    print(f"- Reasoning styles used: {len(set(c['style'] for c in chains))}")
    print(f"- Languages covered: {len(set(c['language'] for c in multilingual_chains))}")
    print(f"- Overall diversity score: {diversity_metrics['overall_diversity']:.3f}")
    
    return {
        'chains': chains,
        'aligned_chains': aligned_chains,
        'multilingual_chains': multilingual_chains,
        'diversity_metrics': diversity_metrics,
        'embeddings': embeddings
    }

def main():
    """Run all DCoTAgentAligner demonstrations"""
    print("DCoTAgentAligner Demonstration Suite")
    print("=" * 50)
    
    try:
        # Run individual demos
        demo_basic_chain_generation()
        demo_clustering_and_visualization()
        demo_graph_alignment()
        demo_diversity_evaluation()
        demo_multilingual_chains()
        
        # Run complete pipeline
        results = demo_complete_pipeline()
        
        print("\n" + "=" * 50)
        print("All demonstrations completed successfully!")
        print("DCoTAgentAligner is working correctly.")
        
        return results
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = main()