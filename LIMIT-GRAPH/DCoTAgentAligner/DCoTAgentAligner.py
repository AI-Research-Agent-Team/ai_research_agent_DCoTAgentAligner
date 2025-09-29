# -*- coding: utf-8 -*-
"""
DCoTAgentAligner: Diverse Chain-of-Thought Agent Alignment Module

This module implements D-CoT (Diverse Chain-of-Thought) patterns for training agents
to generate, cluster, and evaluate diverse reasoning paths across multilingual corpora,
aligning them with semantic graph structures.
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
try:
    from .embed_chain import embed_chain
    from .extract_graph_nodes import extract_graph_nodes
    from .compute_diversity import compute_diversity
except ImportError:
    # Fallback for direct execution
    from embed_chain import embed_chain
    from extract_graph_nodes import extract_graph_nodes
    from compute_diversity import compute_diversity
import logging

logger = logging.getLogger(__name__)

class DCoTAgentAligner:
    """
    Diverse Chain-of-Thought Agent Aligner for multilingual reasoning alignment.
    
    This class orchestrates the generation, clustering, and evaluation of diverse
    reasoning chains across multiple languages, aligning them with semantic graph
    structures for enhanced cognitive diversity.
    """
    
    def __init__(self, model, lang: str = 'id', graph_vocab: Optional[Dict] = None):
        """
        Initialize the DCoTAgentAligner.
        
        Args:
            model: Language model for chain generation
            lang: Primary language code (default: 'id' for Indonesian)
            graph_vocab: Semantic graph vocabulary for alignment
        """
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
        
    def generate_chains(self, prompt: str, n: int = 5, 
                       diverse_styles: bool = True) -> List[Dict[str, Any]]:
        """
        Generate diverse reasoning chains with different cognitive styles.
        
        Args:
            prompt: Input prompt for reasoning
            n: Number of chains to generate
            diverse_styles: Whether to use different reasoning styles
            
        Returns:
            List of reasoning chain dictionaries with metadata
        """
        chains = []
        
        for i in range(n):
            # Apply temperature variation for diversity
            temperature = 0.7 + (i * 0.1)
            
            # Select reasoning style
            style = self.reasoning_styles[i % len(self.reasoning_styles)] if diverse_styles else 'analytical'
            
            # Create style-specific prompt
            styled_prompt = self._create_styled_prompt(prompt, style)
            
            try:
                # Generate chain with model
                generated_text = self.model.generate(
                    styled_prompt, 
                    temperature=temperature,
                    max_tokens=512
                )
                
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

    def cluster_chains(self, chains: List[Dict[str, Any]], 
                      n_clusters: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """
        Cluster reasoning chains based on semantic similarity.
        
        Args:
            chains: List of reasoning chain dictionaries
            n_clusters: Number of clusters to create
            
        Returns:
            Tuple of (cluster_labels, embeddings)
        """
        # Extract text and generate embeddings
        texts = [chain['text'] for chain in chains]
        embeddings = np.array([embed_chain(text) for text in texts])
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # Add cluster info to chains
        for i, chain in enumerate(chains):
            chain['cluster'] = int(cluster_labels[i])
            
        return cluster_labels, embeddings

    def align_to_graph(self, chains: List[Dict[str, Any]], 
                      graph_vocab: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Align reasoning chains to semantic graph structures.
        
        Args:
            chains: List of reasoning chain dictionaries
            graph_vocab: Graph vocabulary for alignment (optional)
            
        Returns:
            List of chains with graph alignment information
        """
        vocab = graph_vocab or self.graph_vocab
        aligned_chains = []
        
        for chain in chains:
            # Extract graph nodes from chain text
            graph_nodes = extract_graph_nodes(chain['text'], vocab)
            
            # Calculate alignment score
            alignment_score = len(graph_nodes) / max(len(vocab), 1)
            
            # Create aligned chain data
            aligned_chain = chain.copy()
            aligned_chain.update({
                'graph_nodes': graph_nodes,
                'alignment_score': alignment_score,
                'graph_coverage': len(set(graph_nodes)) / max(len(vocab), 1)
            })
            
            aligned_chains.append(aligned_chain)
            
        return aligned_chains

    def visualize_clusters(self, chains: List[Dict[str, Any]], 
                          embeddings: np.ndarray, 
                          save_path: Optional[str] = None) -> None:
        """
        Visualize reasoning chain clusters using t-SNE.
        
        Args:
            chains: List of reasoning chain dictionaries
            embeddings: Chain embeddings for visualization
            save_path: Optional path to save the plot
        """
        # Reduce dimensionality for visualization
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings)
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        # Get cluster labels and styles
        clusters = [chain.get('cluster', 0) for chain in chains]
        styles = [chain.get('style', 'unknown') for chain in chains]
        
        # Create scatter plot
        scatter = plt.scatter(
            embeddings_2d[:, 0], 
            embeddings_2d[:, 1],
            c=clusters, 
            cmap='viridis',
            alpha=0.7,
            s=100
        )
        
        # Add style annotations
        for i, (x, y, style) in enumerate(zip(embeddings_2d[:, 0], 
                                            embeddings_2d[:, 1], 
                                            styles)):
            plt.annotate(f'{i}:{style[:3]}', (x, y), 
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.8)
        
        plt.colorbar(scatter, label='Cluster')
        plt.title('D-CoT Reasoning Chain Clusters')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def evaluate_diversity(self, chains: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Evaluate the diversity of reasoning chains.
        
        Args:
            chains: List of reasoning chain dictionaries
            
        Returns:
            Dictionary with diversity metrics
        """
        texts = [chain['text'] for chain in chains]
        
        # Compute semantic diversity
        semantic_diversity = compute_diversity(texts)
        
        # Compute style diversity
        styles = [chain.get('style', 'unknown') for chain in chains]
        style_diversity = len(set(styles)) / len(styles)
        
        # Compute language diversity (if multilingual)
        languages = [chain.get('language', self.lang) for chain in chains]
        language_diversity = len(set(languages)) / len(languages)
        
        return {
            'semantic_diversity': semantic_diversity,
            'style_diversity': style_diversity,
            'language_diversity': language_diversity,
            'overall_diversity': (semantic_diversity + style_diversity + language_diversity) / 3
        }

    def generate_multilingual_chains(self, prompt: str, 
                                   languages: List[str] = None,
                                   chains_per_lang: int = 3) -> List[Dict[str, Any]]:
        """
        Generate reasoning chains across multiple languages.
        
        Args:
            prompt: Base prompt for reasoning
            languages: List of language codes
            chains_per_lang: Number of chains per language
            
        Returns:
            List of multilingual reasoning chains
        """
        languages = languages or ['id', 'en', 'zh']
        all_chains = []
        
        for lang in languages:
            if lang in self.multilingual_prompts:
                lang_prompt = f"{self.multilingual_prompts[lang]} {prompt}"
                
                # Temporarily set language
                original_lang = self.lang
                self.lang = lang
                
                # Generate chains for this language
                lang_chains = self.generate_chains(
                    lang_prompt, 
                    n=chains_per_lang,
                    diverse_styles=True
                )
                
                # Add language metadata
                for chain in lang_chains:
                    chain['language'] = lang
                    
                all_chains.extend(lang_chains)
                
                # Restore original language
                self.lang = original_lang
                
        return all_chains

    def _create_styled_prompt(self, prompt: str, style: str) -> str:
        """
        Create a style-specific prompt for diverse reasoning.
        
        Args:
            prompt: Base prompt
            style: Reasoning style
            
        Returns:
            Styled prompt string
        """
        style_prefixes = {
            'analytical': 'Using systematic analysis and logical reasoning:',
            'creative': 'Using creative and innovative thinking:',
            'systematic': 'Using a structured, step-by-step approach:',
            'intuitive': 'Using intuitive insights and pattern recognition:',
            'critical': 'Using critical evaluation and questioning:'
        }
        
        prefix = style_prefixes.get(style, 'Using careful reasoning:')
        return f"{prefix} {prompt}"
