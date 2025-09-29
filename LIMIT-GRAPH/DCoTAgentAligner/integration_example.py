# -*- coding: utf-8 -*-
"""
Integration example showing how to use DCoTAgentAligner with the main AI research agent
"""

import sys
import os
import json
from typing import Dict, List, Any

# Add paths for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

class ResearchAgentLLMWrapper:
    """
    Wrapper to integrate DCoTAgentAligner with the main research agent's LLM
    """
    
    def __init__(self, research_agent=None):
        self.research_agent = research_agent
        self.fallback_responses = {
            'analytical': "Analyzing this systematically: First, I'll examine the core components, then evaluate relationships, and finally synthesize conclusions based on evidence.",
            'creative': "Approaching this creatively: Let me explore unconventional angles, consider novel combinations, and think beyond traditional boundaries.",
            'systematic': "Using a systematic approach: Step 1 - Define objectives, Step 2 - Gather information, Step 3 - Process methodically, Step 4 - Validate results.",
            'intuitive': "Following intuitive reasoning: I sense patterns emerging that suggest deeper connections and relationships worth exploring further.",
            'critical': "Applying critical analysis: What assumptions underlie this? What evidence supports or contradicts? What alternative explanations exist?"
        }
    
    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 512) -> str:
        """Generate response using research agent or fallback"""
        if self.research_agent and hasattr(self.research_agent, 'llm'):
            try:
                # Use the research agent's LLM
                response = self.research_agent.llm.invoke(prompt)
                return response.content if hasattr(response, 'content') else str(response)
            except Exception as e:
                print(f"Research agent LLM failed: {e}")
        
        # Fallback to mock responses
        style = self._extract_style(prompt)
        return self.fallback_responses.get(style, "Analyzing the given prompt and providing a structured response.")
    
    def _extract_style(self, prompt: str) -> str:
        """Extract reasoning style from prompt"""
        prompt_lower = prompt.lower()
        for style in self.fallback_responses.keys():
            if style in prompt_lower:
                return style
        return 'analytical'

def create_research_graph_vocab() -> Dict[str, List[str]]:
    """Create research-specific graph vocabulary"""
    return {
        'research_concepts': [
            'hypothesis', 'evidence', 'analysis', 'methodology', 'validation',
            'experiment', 'observation', 'theory', 'model', 'framework'
        ],
        'cognitive_processes': [
            'reasoning', 'inference', 'deduction', 'induction', 'synthesis',
            'evaluation', 'comparison', 'classification', 'abstraction'
        ],
        'research_methods': [
            'survey', 'interview', 'case_study', 'experiment', 'observation',
            'literature_review', 'meta_analysis', 'statistical_analysis'
        ],
        'knowledge_relations': [
            'causes', 'correlates_with', 'leads_to', 'supports', 'contradicts',
            'explains', 'predicts', 'influences', 'depends_on', 'enables'
        ]
    }

def demonstrate_research_integration():
    """Demonstrate DCoTAgentAligner integration with research workflows"""
    print("DCoTAgentAligner Research Integration Demo")
    print("=" * 60)
    
    # Import the standalone implementation
    from standalone_demo import DCoTAgentAligner
    
    # Initialize with research-specific configuration
    research_llm = ResearchAgentLLMWrapper()
    research_vocab = create_research_graph_vocab()
    
    aligner = DCoTAgentAligner(
        model=research_llm,
        lang='en',
        graph_vocab=research_vocab
    )
    
    # Research scenario: Analyzing a complex research question
    research_question = "How can we improve the reliability of AI systems in critical applications?"
    
    print(f"\nResearch Question: {research_question}")
    print("-" * 60)
    
    # Generate diverse reasoning approaches
    print("\n1. Generating diverse research approaches...")
    research_chains = aligner.generate_chains(
        research_question,
        n=6,
        diverse_styles=True
    )
    
    for i, chain in enumerate(research_chains):
        print(f"\nApproach {i+1} ({chain['style'].upper()}):")
        print(f"  {chain['text'][:120]}...")
    
    # Analyze alignment with research concepts
    print("\n2. Analyzing alignment with research concepts...")
    aligned_chains = aligner.align_to_graph(research_chains, research_vocab)
    
    for i, chain in enumerate(aligned_chains):
        print(f"\nChain {i+1} Research Alignment:")
        print(f"  Concepts found: {chain['graph_nodes'][:5]}...")  # Show first 5
        print(f"  Alignment score: {chain['alignment_score']:.3f}")
        print(f"  Research coverage: {chain['graph_coverage']:.3f}")
    
    # Evaluate methodological diversity
    print("\n3. Evaluating methodological diversity...")
    diversity_metrics = aligner.evaluate_diversity(aligned_chains)
    
    print("Research Diversity Analysis:")
    for metric, value in diversity_metrics.items():
        print(f"  {metric.replace('_', ' ').title()}: {value:.3f}")
    
    # Generate cross-cultural research perspectives
    print("\n4. Generating cross-cultural research perspectives...")
    multicultural_chains = aligner.generate_multilingual_chains(
        research_question,
        languages=['en', 'id', 'zh'],
        chains_per_lang=2
    )
    
    print(f"Generated {len(multicultural_chains)} cross-cultural perspectives:")
    for i, chain in enumerate(multicultural_chains):
        print(f"  Perspective {i+1} ({chain['language']}): {chain['text'][:80]}...")
    
    return {
        'research_chains': research_chains,
        'aligned_chains': aligned_chains,
        'diversity_metrics': diversity_metrics,
        'multicultural_chains': multicultural_chains
    }

def export_research_results(results: Dict[str, Any], filename: str = "dcot_research_analysis.json"):
    """Export research results for further analysis"""
    
    # Prepare data for export
    export_data = {
        'metadata': {
            'analysis_type': 'DCoT Research Integration',
            'timestamp': '2024-01-01',  # Would use actual timestamp
            'total_chains': len(results.get('research_chains', [])),
            'languages_analyzed': len(set(c.get('language', 'en') for c in results.get('multicultural_chains', []))),
            'reasoning_styles': len(set(c.get('style', 'analytical') for c in results.get('research_chains', [])))
        },
        'diversity_analysis': results.get('diversity_metrics', {}),
        'research_approaches': [
            {
                'id': chain['id'],
                'style': chain['style'],
                'alignment_score': chain.get('alignment_score', 0),
                'graph_coverage': chain.get('graph_coverage', 0),
                'text_preview': chain['text'][:200] + "..." if len(chain['text']) > 200 else chain['text']
            }
            for chain in results.get('aligned_chains', [])
        ],
        'cross_cultural_perspectives': [
            {
                'language': chain['language'],
                'style': chain['style'],
                'text_preview': chain['text'][:150] + "..." if len(chain['text']) > 150 else chain['text']
            }
            for chain in results.get('multicultural_chains', [])
        ]
    }
    
    # Save to file
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nResearch analysis exported to: {filename}")
    return export_data

def generate_research_report(results: Dict[str, Any]) -> str:
    """Generate a comprehensive research report"""
    
    diversity = results.get('diversity_metrics', {})
    chains = results.get('aligned_chains', [])
    multicultural = results.get('multicultural_chains', [])
    
    report = f"""
# DCoT Research Analysis Report

## Executive Summary

This analysis employed Diverse Chain-of-Thought (D-CoT) methodology to examine research approaches across multiple cognitive styles and cultural perspectives.

## Key Findings

### Methodological Diversity
- **Overall Diversity Score**: {diversity.get('overall_diversity', 0):.3f}
- **Cognitive Style Diversity**: {diversity.get('style_diversity', 0):.3f}
- **Semantic Diversity**: {diversity.get('semantic_diversity', 0):.3f}
- **Cross-Cultural Coverage**: {diversity.get('language_diversity', 0):.3f}

### Research Approach Analysis

Generated {len(chains)} distinct research approaches across {len(set(c.get('style', 'analytical') for c in chains))} cognitive styles:

"""
    
    # Add approach summaries
    for i, chain in enumerate(chains[:3]):  # Top 3 approaches
        report += f"""
#### Approach {i+1}: {chain.get('style', 'Unknown').title()} Method
- **Research Alignment**: {chain.get('alignment_score', 0):.3f}
- **Concept Coverage**: {chain.get('graph_coverage', 0):.3f}
- **Key Concepts**: {', '.join(chain.get('graph_nodes', [])[:5])}

"""
    
    # Add cross-cultural insights
    if multicultural:
        report += f"""
### Cross-Cultural Research Perspectives

Analyzed {len(multicultural)} perspectives across {len(set(c.get('language', 'en') for c in multicultural))} cultural contexts:

"""
        
        for lang in set(c.get('language', 'en') for c in multicultural):
            lang_chains = [c for c in multicultural if c.get('language') == lang]
            report += f"- **{lang.upper()}**: {len(lang_chains)} perspectives generated\n"
    
    report += """
## Recommendations

1. **Methodological Integration**: Combine analytical and creative approaches for comprehensive research
2. **Cross-Cultural Validation**: Validate findings across multiple cultural contexts
3. **Diverse Reasoning**: Employ multiple cognitive styles to avoid methodological bias
4. **Semantic Alignment**: Ensure research approaches align with established conceptual frameworks

## Conclusion

The D-CoT analysis reveals significant diversity in research approaches, suggesting the value of multi-perspective methodologies for robust research outcomes.
"""
    
    return report

def main():
    """Run the complete research integration demo"""
    try:
        # Run the integration demo
        results = demonstrate_research_integration()
        
        # Export results
        export_data = export_research_results(results)
        
        # Generate report
        report = generate_research_report(results)
        
        # Save report
        with open('dcot_research_report.md', 'w', encoding='utf-8') as f:
            f.write(report)
        
        print("\n" + "=" * 60)
        print("Research integration demo completed successfully!")
        print("Files generated:")
        print("  - dcot_research_analysis.json (detailed data)")
        print("  - dcot_research_report.md (summary report)")
        
        return results
        
    except Exception as e:
        print(f"Integration demo failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = main()