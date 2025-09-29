# DCoTAgentAligner: Diverse Chain-of-Thought Agent Alignment

A sophisticated system for generating, clustering, and evaluating diverse reasoning chains across multiple languages and cognitive styles, with semantic graph alignment capabilities.

## Overview

DCoTAgentAligner implements Diverse Chain-of-Thought (D-CoT) patterns for training agents to:
- Generate reasoning chains with different cognitive styles
- Cluster and analyze semantic similarity
- Align reasoning with semantic graph structures
- Support multilingual reasoning generation
- Evaluate diversity and consistency metrics

## Features

### 🧠 Diverse Reasoning Generation
- **Multiple Cognitive Styles**: Analytical, creative, systematic, intuitive, and critical thinking
- **Temperature Variation**: Automatic temperature adjustment for diversity
- **Style-Specific Prompting**: Tailored prompts for each reasoning style

### 🌐 Multilingual Support
- **Language Coverage**: Indonesian, English, Chinese, Spanish, Arabic
- **Cross-lingual Alignment**: Consistent reasoning across languages
- **Cultural Adaptation**: Language-specific reasoning patterns

### 📊 Advanced Analytics
- **Semantic Clustering**: K-means clustering with t-SNE visualization
- **Diversity Metrics**: Semantic, style, and language diversity scores
- **Graph Alignment**: Alignment with semantic graph vocabularies
- **Performance Evaluation**: Comprehensive validation framework

### 🔗 Graph Integration
- **Semantic Alignment**: Map reasoning chains to graph concepts
- **Coverage Analysis**: Measure graph vocabulary coverage
- **Relationship Extraction**: Identify conceptual relationships

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or install individual packages
pip install numpy scikit-learn sentence-transformers matplotlib seaborn
```

## Quick Start

```python
from DCoTAgentAligner import DCoTAgentAligner

# Initialize with your language model
aligner = DCoTAgentAligner(
    model=your_llm_model,
    lang='en',
    graph_vocab=your_graph_vocabulary
)

# Generate diverse reasoning chains
prompt = "How can we solve complex problems effectively?"
chains = aligner.generate_chains(prompt, n=5, diverse_styles=True)

# Cluster and analyze
cluster_labels, embeddings = aligner.cluster_chains(chains, n_clusters=3)

# Align with semantic graph
aligned_chains = aligner.align_to_graph(chains)

# Evaluate diversity
diversity_metrics = aligner.evaluate_diversity(aligned_chains)
```

## Core Components

### DCoTAgentAligner Class

The main orchestrator class that coordinates all D-CoT operations:

```python
class DCoTAgentAligner:
    def __init__(self, model, lang='id', graph_vocab=None)
    def generate_chains(self, prompt, n=5, diverse_styles=True)
    def cluster_chains(self, chains, n_clusters=3)
    def align_to_graph(self, chains, graph_vocab=None)
    def evaluate_diversity(self, chains)
    def generate_multilingual_chains(self, prompt, languages=None)
    def visualize_clusters(self, chains, embeddings, save_path=None)
```

### Supporting Modules

- **embed_chain.py**: Semantic embedding generation using SentenceTransformers
- **extract_graph_nodes.py**: Graph concept extraction from reasoning text
- **compute_diversity.py**: Diversity metrics calculation using TF-IDF and cosine similarity
- **validate_dcot_submission.py**: Comprehensive validation framework

## Usage Examples

### Basic Chain Generation

```python
# Generate chains with different reasoning styles
chains = aligner.generate_chains(
    "Explain the scientific method",
    n=5,
    diverse_styles=True
)

for chain in chains:
    print(f"Style: {chain['style']}")
    print(f"Text: {chain['text'][:100]}...")
```

### Multilingual Generation

```python
# Generate reasoning in multiple languages
multilingual_chains = aligner.generate_multilingual_chains(
    "problem solving methodology",
    languages=['en', 'id', 'zh'],
    chains_per_lang=3
)
```

### Clustering and Visualization

```python
# Cluster chains and visualize
cluster_labels, embeddings = aligner.cluster_chains(chains, n_clusters=4)
aligner.visualize_clusters(chains, embeddings, save_path='clusters.png')
```

### Graph Alignment

```python
# Define graph vocabulary
graph_vocab = {
    'concepts': ['analysis', 'reasoning', 'logic'],
    'relations': ['causes', 'leads_to'],
    'entities': ['problem', 'solution']
}

# Align chains to graph
aligned_chains = aligner.align_to_graph(chains, graph_vocab)

for chain in aligned_chains:
    print(f"Graph nodes: {chain['graph_nodes']}")
    print(f"Alignment score: {chain['alignment_score']:.3f}")
```

### Diversity Evaluation

```python
# Evaluate reasoning diversity
diversity_metrics = aligner.evaluate_diversity(chains)

print(f"Semantic diversity: {diversity_metrics['semantic_diversity']:.3f}")
print(f"Style diversity: {diversity_metrics['style_diversity']:.3f}")
print(f"Overall diversity: {diversity_metrics['overall_diversity']:.3f}")
```

## Validation Framework

The system includes comprehensive validation for D-CoT submissions:

```python
from validate_dcot_submission import validate_dcot_submission

# Validate a submission file
results = validate_dcot_submission('submission.json', lang='en')

print(f"Diversity score: {results['diversity']:.3f}")
print(f"Graph consistency: {results['graph_consistency']:.3f}")
print(f"Multilingual accuracy: {results['multilingual_accuracy']:.3f}")
print(f"Overall score: {results['overall_score']:.3f}")
```

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
python test_dcot_system.py

# Run demo
python demo_dcot_aligner.py
```

## Configuration

### Reasoning Styles

The system supports five cognitive reasoning styles:

- **Analytical**: Systematic analysis and logical reasoning
- **Creative**: Innovative and unconventional thinking
- **Systematic**: Structured, step-by-step approaches
- **Intuitive**: Pattern recognition and insight-based reasoning
- **Critical**: Questioning and evaluation-focused thinking

### Multilingual Prompts

Customize prompts for different languages:

```python
multilingual_prompts = {
    'id': 'Jelaskan langkah demi langkah:',
    'en': 'Explain step by step:',
    'zh': '逐步解释：',
    'es': 'Explica paso a paso:',
    'ar': 'اشرح خطوة بخطوة:'
}
```

### Graph Vocabulary

Define semantic graph structures:

```python
graph_vocab = {
    'concepts': ['analysis', 'reasoning', 'logic', 'creativity'],
    'relations': ['causes', 'leads_to', 'requires', 'supports'],
    'entities': ['problem', 'solution', 'method', 'result']
}
```

## Performance Metrics

The system provides comprehensive evaluation metrics:

- **Semantic Diversity**: Measures conceptual variation using TF-IDF cosine similarity
- **Style Diversity**: Evaluates cognitive style distribution
- **Language Diversity**: Assesses multilingual coverage
- **Graph Alignment**: Measures semantic graph consistency
- **Overall Score**: Composite metric combining all dimensions

## Integration with LIMIT-GRAPH

DCoTAgentAligner integrates seamlessly with the broader LIMIT-GRAPH ecosystem:

- **Semantic Graph Pipeline**: Feeds into graph construction workflows
- **Agent Training**: Provides diverse training data for reasoning agents
- **Evaluation Harness**: Contributes to benchmark evaluation systems
- **Multilingual Support**: Enhances cross-lingual capabilities

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is part of the LIMIT-GRAPH research framework and follows the same licensing terms.

## Citation

If you use DCoTAgentAligner in your research, please cite:

```bibtex
@software{dcot_agent_aligner,
  title={DCoTAgentAligner: Diverse Chain-of-Thought Agent Alignment},
  author={LIMIT-GRAPH Research Team},
  year={2024},
  url={https://github.com/your-repo/LIMIT-GRAPH}
}
```