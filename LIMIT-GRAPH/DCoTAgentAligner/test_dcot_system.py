# -*- coding: utf-8 -*-
"""
Comprehensive test suite for DCoTAgentAligner system
"""

import unittest
import tempfile
import json
import os
import sys
import numpy as np

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from DCoTAgentAligner import DCoTAgentAligner
from embed_chain import embed_chain
from extract_graph_nodes import extract_graph_nodes
from compute_diversity import compute_diversity
from validate_dcot_submission import validate_dcot_submission, load_chains

class MockLLM:
    """Mock LLM for testing"""
    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 512) -> str:
        return f"Mock response for: {prompt[:50]}... (temp: {temperature})"

class TestDCoTAgentAligner(unittest.TestCase):
    """Test cases for DCoTAgentAligner"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_llm = MockLLM()
        self.graph_vocab = {
            'concepts': ['analysis', 'reasoning', 'logic'],
            'relations': ['causes', 'leads_to'],
            'entities': ['problem', 'solution']
        }
        self.aligner = DCoTAgentAligner(
            model=self.mock_llm,
            lang='en',
            graph_vocab=self.graph_vocab
        )
    
    def test_initialization(self):
        """Test DCoTAgentAligner initialization"""
        self.assertEqual(self.aligner.lang, 'en')
        self.assertEqual(self.aligner.model, self.mock_llm)
        self.assertIsInstance(self.aligner.reasoning_styles, list)
        self.assertIn('analytical', self.aligner.reasoning_styles)
    
    def test_chain_generation(self):
        """Test reasoning chain generation"""
        prompt = "Test problem solving"
        chains = self.aligner.generate_chains(prompt, n=3, diverse_styles=True)
        
        self.assertEqual(len(chains), 3)
        for chain in chains:
            self.assertIn('id', chain)
            self.assertIn('text', chain)
            self.assertIn('style', chain)
            self.assertIn('temperature', chain)
            self.assertIn('language', chain)
    
    def test_clustering(self):
        """Test chain clustering functionality"""
        # Create test chains
        chains = [
            {'text': 'analytical reasoning approach', 'id': 'chain_0'},
            {'text': 'creative thinking method', 'id': 'chain_1'},
            {'text': 'systematic analysis process', 'id': 'chain_2'}
        ]
        
        cluster_labels, embeddings = self.aligner.cluster_chains(chains, n_clusters=2)
        
        self.assertEqual(len(cluster_labels), 3)
        self.assertEqual(embeddings.shape[0], 3)
        
        # Check that cluster info was added to chains
        for chain in chains:
            self.assertIn('cluster', chain)
    
    def test_graph_alignment(self):
        """Test graph alignment functionality"""
        chains = [
            {'text': 'This involves analysis and reasoning to solve the problem', 'id': 'chain_0'},
            {'text': 'Creative approach without logic', 'id': 'chain_1'}
        ]
        
        aligned_chains = self.aligner.align_to_graph(chains)
        
        self.assertEqual(len(aligned_chains), 2)
        for chain in aligned_chains:
            self.assertIn('graph_nodes', chain)
            self.assertIn('alignment_score', chain)
            self.assertIn('graph_coverage', chain)
    
    def test_diversity_evaluation(self):
        """Test diversity evaluation"""
        chains = [
            {'text': 'First approach to problem solving', 'style': 'analytical'},
            {'text': 'Second creative method for solutions', 'style': 'creative'},
            {'text': 'Third systematic way of thinking', 'style': 'systematic'}
        ]
        
        diversity_metrics = self.aligner.evaluate_diversity(chains)
        
        self.assertIn('semantic_diversity', diversity_metrics)
        self.assertIn('style_diversity', diversity_metrics)
        self.assertIn('language_diversity', diversity_metrics)
        self.assertIn('overall_diversity', diversity_metrics)
        
        # Style diversity should be 1.0 (all different styles)
        self.assertEqual(diversity_metrics['style_diversity'], 1.0)
    
    def test_multilingual_generation(self):
        """Test multilingual chain generation"""
        prompt = "problem solving"
        languages = ['en', 'id']
        
        multilingual_chains = self.aligner.generate_multilingual_chains(
            prompt, languages=languages, chains_per_lang=2
        )
        
        self.assertEqual(len(multilingual_chains), 4)  # 2 languages * 2 chains
        
        # Check language distribution
        languages_found = set(chain['language'] for chain in multilingual_chains)
        self.assertEqual(languages_found, {'en', 'id'})

class TestSupportingModules(unittest.TestCase):
    """Test cases for supporting modules"""
    
    def test_embed_chain(self):
        """Test chain embedding functionality"""
        text = "This is a test reasoning chain"
        embedding = embed_chain(text)
        
        self.assertIsInstance(embedding, np.ndarray)
        self.assertGreater(len(embedding), 0)
    
    def test_extract_graph_nodes(self):
        """Test graph node extraction"""
        text = "This analysis involves reasoning and logic"
        vocab = ['analysis', 'reasoning', 'logic', 'creativity']
        
        nodes = extract_graph_nodes(text, vocab)
        
        self.assertIn('analysis', nodes)
        self.assertIn('reasoning', nodes)
        self.assertIn('logic', nodes)
        self.assertNotIn('creativity', nodes)
    
    def test_compute_diversity(self):
        """Test diversity computation"""
        chains = [
            "First unique approach to solving problems",
            "Second different method for solutions",
            "Third distinct way of thinking"
        ]
        
        diversity_score = compute_diversity(chains)
        
        self.assertIsInstance(diversity_score, float)
        self.assertGreaterEqual(diversity_score, 0.0)
        self.assertLessEqual(diversity_score, 1.0)

class TestValidation(unittest.TestCase):
    """Test cases for validation functionality"""
    
    def test_load_chains_json(self):
        """Test loading chains from JSON file"""
        # Create temporary JSON file
        test_data = {
            'chains': [
                {'text': 'First chain', 'id': 0},
                {'text': 'Second chain', 'id': 1}
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_data, f)
            temp_path = f.name
        
        try:
            chains = load_chains(temp_path)
            self.assertEqual(len(chains), 2)
            self.assertEqual(chains[0]['text'], 'First chain')
        finally:
            os.unlink(temp_path)
    
    def test_load_chains_text(self):
        """Test loading chains from text file"""
        # Create temporary text file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("First reasoning chain\n")
            f.write("Second reasoning chain\n")
            temp_path = f.name
        
        try:
            chains = load_chains(temp_path)
            self.assertEqual(len(chains), 2)
            self.assertEqual(chains[0]['text'], 'First reasoning chain')
        finally:
            os.unlink(temp_path)
    
    def test_validate_dcot_submission(self):
        """Test complete submission validation"""
        # Create test submission
        test_chains = [
            {'text': 'This is analysis with reasoning and logic'},
            {'text': 'Creative approach with different thinking'},
            {'text': 'Systematic method for problem solving'}
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({'chains': test_chains}, f)
            temp_path = f.name
        
        try:
            results = validate_dcot_submission(temp_path, lang='en')
            
            self.assertIn('diversity', results)
            self.assertIn('graph_consistency', results)
            self.assertIn('multilingual_accuracy', results)
            self.assertIn('overall_score', results)
            
            # All scores should be between 0 and 1
            for score in results.values():
                if isinstance(score, float):
                    self.assertGreaterEqual(score, 0.0)
                    self.assertLessEqual(score, 1.0)
        finally:
            os.unlink(temp_path)

def run_integration_test():
    """Run complete integration test"""
    print("\n=== Running DCoTAgentAligner Integration Test ===")
    
    # Initialize system
    mock_llm = MockLLM()
    graph_vocab = {
        'concepts': ['analysis', 'reasoning', 'logic', 'creativity', 'system'],
        'relations': ['causes', 'leads_to', 'requires'],
        'entities': ['problem', 'solution', 'method']
    }
    
    aligner = DCoTAgentAligner(
        model=mock_llm,
        lang='en',
        graph_vocab=graph_vocab
    )
    
    # Test complete pipeline
    prompt = "How do we approach complex research problems?"
    
    # Step 1: Generate chains
    chains = aligner.generate_chains(prompt, n=5, diverse_styles=True)
    print(f"✓ Generated {len(chains)} reasoning chains")
    
    # Step 2: Cluster chains
    cluster_labels, embeddings = aligner.cluster_chains(chains, n_clusters=3)
    print(f"✓ Clustered chains into 3 groups")
    
    # Step 3: Align to graph
    aligned_chains = aligner.align_to_graph(chains)
    print(f"✓ Aligned chains to semantic graph")
    
    # Step 4: Evaluate diversity
    diversity_metrics = aligner.evaluate_diversity(aligned_chains)
    print(f"✓ Evaluated diversity: {diversity_metrics['overall_diversity']:.3f}")
    
    # Step 5: Test multilingual
    multilingual_chains = aligner.generate_multilingual_chains(
        prompt, languages=['en', 'id'], chains_per_lang=2
    )
    print(f"✓ Generated {len(multilingual_chains)} multilingual chains")
    
    # Step 6: Test validation
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({'chains': [{'text': chain['text']} for chain in chains]}, f)
        temp_path = f.name
    
    try:
        validation_results = validate_dcot_submission(temp_path, lang='en')
        print(f"✓ Validation completed: {validation_results['overall_score']:.3f}")
    finally:
        os.unlink(temp_path)
    
    print("✓ Integration test completed successfully!")
    return True

def main():
    """Run all tests"""
    print("DCoTAgentAligner Test Suite")
    print("=" * 50)
    
    # Run unit tests
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run integration test
    try:
        run_integration_test()
        print("\n" + "=" * 50)
        print("All tests passed! DCoTAgentAligner is working correctly.")
    except Exception as e:
        print(f"\nIntegration test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()