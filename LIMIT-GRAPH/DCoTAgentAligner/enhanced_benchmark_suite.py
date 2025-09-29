# -*- coding: utf-8 -*-
"""
Enhanced Benchmarking Suite for DCoTAgentAligner

Advanced benchmarking with improved performance, sophisticated metrics,
statistical analysis, and comprehensive evaluation capabilities.
"""

import sys
import os
import json
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from dataclasses import dataclass, asdict
from collections import defaultdict
import statistics
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

# Import our modules
from standalone_demo import DCoTAgentAligner, MockLLM

@dataclass
class BenchmarkMetrics:
    """Structured benchmark metrics with statistical analysis"""
    cognitive_diversity: float
    multilingual_alignment: float
    graph_consistency: float
    performance_score: float
    composite_score: float
    execution_time: float
    confidence_interval: Tuple[float, float]
    statistical_significance: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class AdvancedBenchmarkConfig:
    """Advanced benchmark configuration"""
    name: str
    reasoning_styles: List[str]
    languages: List[str]
    graph_vocab_size: int
    test_iterations: int = 5
    parallel_execution: bool = True
    statistical_analysis: bool = True
    confidence_level: float = 0.95

class StatisticalAnalyzer:
    """Advanced statistical analysis for benchmark results"""
    
    @staticmethod
    def compute_confidence_interval(data: List[float], confidence: float = 0.95) -> Tuple[float, float]:
        """Compute confidence interval for benchmark scores"""
        if len(data) < 2:
            return (0.0, 0.0)
        
        mean = np.mean(data)
        std_err = statistics.stdev(data) / np.sqrt(len(data))
        
        # Use t-distribution for small samples
        from scipy import stats
        t_value = stats.t.ppf((1 + confidence) / 2, len(data) - 1)
        margin = t_value * std_err
        
        return (mean - margin, mean + margin)
    
    @staticmethod
    def compute_effect_size(group1: List[float], group2: List[float]) -> float:
        """Compute Cohen's d effect size between two groups"""
        if len(group1) < 2 or len(group2) < 2:
            return 0.0
        
        mean1, mean2 = np.mean(group1), np.mean(group2)
        std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
        
        # Pooled standard deviation
        n1, n2 = len(group1), len(group2)
        pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
        
        return (mean1 - mean2) / pooled_std if pooled_std > 0 else 0.0
    
    @staticmethod
    def perform_significance_test(group1: List[float], group2: List[float]) -> float:
        """Perform statistical significance test (t-test)"""
        try:
            from scipy import stats
            t_stat, p_value = stats.ttest_ind(group1, group2)
            return p_value
        except ImportError:
            # Fallback: simple difference test
            mean1, mean2 = np.mean(group1), np.mean(group2)
            return abs(mean1 - mean2)

class EnhancedCognitiveDiversityBenchmark:
    """Enhanced cognitive diversity benchmark with advanced metrics"""
    
    def __init__(self):
        self.style_complexity_weights = {
            'analytical': 1.0,
            'creative': 1.2,
            'systematic': 0.9,
            'intuitive': 1.1,
            'critical': 1.3
        }
    
    def compute_advanced_diversity_metrics(self, chains: List[Dict[str, Any]]) -> Dict[str, float]:
        """Compute advanced cognitive diversity metrics"""
        if not chains:
            return self._empty_metrics()
        
        # Basic diversity
        basic_diversity = self._compute_basic_diversity(chains)
        
        # Style complexity analysis
        complexity_score = self._compute_style_complexity(chains)
        
        # Semantic coherence
        coherence_score = self._compute_semantic_coherence(chains)
        
        # Reasoning depth analysis
        depth_score = self._compute_reasoning_depth(chains)
        
        # Cross-style consistency
        consistency_score = self._compute_cross_style_consistency(chains)
        
        return {
            'basic_diversity': basic_diversity,
            'style_complexity': complexity_score,
            'semantic_coherence': coherence_score,
            'reasoning_depth': depth_score,
            'cross_style_consistency': consistency_score,
            'advanced_diversity_score': np.mean([
                basic_diversity, complexity_score, coherence_score, 
                depth_score, consistency_score
            ])
        }
    
    def _compute_basic_diversity(self, chains: List[Dict[str, Any]]) -> float:
        """Compute basic diversity score"""
        styles = [chain.get('style', 'unknown') for chain in chains]
        unique_styles = set(styles)
        return len(unique_styles) / len(styles) if styles else 0.0
    
    def _compute_style_complexity(self, chains: List[Dict[str, Any]]) -> float:
        """Compute weighted style complexity score"""
        total_weight = 0.0
        total_chains = 0
        
        for chain in chains:
            style = chain.get('style', 'analytical')
            weight = self.style_complexity_weights.get(style, 1.0)
            total_weight += weight
            total_chains += 1
        
        return total_weight / max(total_chains, 1)
    
    def _compute_semantic_coherence(self, chains: List[Dict[str, Any]]) -> float:
        """Compute semantic coherence across reasoning chains"""
        texts = [chain.get('text', '') for chain in chains]
        
        # Simple coherence based on shared vocabulary
        all_words = set()
        word_counts = defaultdict(int)
        
        for text in texts:
            words = set(text.lower().split())
            all_words.update(words)
            for word in words:
                word_counts[word] += 1
        
        # Coherence as ratio of shared vocabulary
        shared_words = sum(1 for count in word_counts.values() if count > 1)
        return shared_words / max(len(all_words), 1)
    
    def _compute_reasoning_depth(self, chains: List[Dict[str, Any]]) -> float:
        """Compute reasoning depth based on text analysis"""
        depth_scores = []
        
        for chain in chains:
            text = chain.get('text', '')
            
            # Depth indicators
            sentence_count = text.count('.') + text.count('!') + text.count('?')
            logical_connectors = sum(text.lower().count(word) for word in [
                'because', 'therefore', 'however', 'moreover', 'furthermore',
                'consequently', 'thus', 'hence', 'since', 'although'
            ])
            
            # Normalize depth score
            depth = (sentence_count + logical_connectors * 2) / max(len(text.split()), 1)
            depth_scores.append(min(depth, 1.0))  # Cap at 1.0
        
        return np.mean(depth_scores) if depth_scores else 0.0
    
    def _compute_cross_style_consistency(self, chains: List[Dict[str, Any]]) -> float:
        """Compute consistency across different reasoning styles"""
        style_groups = defaultdict(list)
        
        for chain in chains:
            style = chain.get('style', 'unknown')
            text_length = len(chain.get('text', ''))
            style_groups[style].append(text_length)
        
        if len(style_groups) < 2:
            return 1.0
        
        # Compute coefficient of variation across styles
        style_means = [np.mean(lengths) for lengths in style_groups.values()]
        cv = np.std(style_means) / np.mean(style_means) if np.mean(style_means) > 0 else 0.0
        
        return max(0.0, 1.0 - cv)  # Lower CV = higher consistency
    
    def _empty_metrics(self) -> Dict[str, float]:
        """Return empty metrics for edge cases"""
        return {
            'basic_diversity': 0.0,
            'style_complexity': 0.0,
            'semantic_coherence': 0.0,
            'reasoning_depth': 0.0,
            'cross_style_consistency': 0.0,
            'advanced_diversity_score': 0.0
        }

class EnhancedMultilingualBenchmark:
    """Enhanced multilingual benchmark with linguistic analysis"""
    
    def __init__(self):
        self.language_complexity = {
            'en': 1.0,  # English baseline
            'id': 1.1,  # Indonesian - agglutinative
            'zh': 1.3,  # Chinese - logographic
            'es': 1.0,  # Spanish - similar to English
            'ar': 1.4   # Arabic - right-to-left, complex morphology
        }
    
    def compute_advanced_multilingual_metrics(self, chains: List[Dict[str, Any]]) -> Dict[str, float]:
        """Compute advanced multilingual alignment metrics"""
        if not chains:
            return self._empty_multilingual_metrics()
        
        # Language distribution analysis
        lang_distribution = self._analyze_language_distribution(chains)
        
        # Cross-lingual semantic consistency
        semantic_consistency = self._compute_cross_lingual_semantics(chains)
        
        # Language complexity handling
        complexity_handling = self._evaluate_complexity_handling(chains)
        
        # Cultural adaptation assessment
        cultural_adaptation = self._assess_cultural_adaptation(chains)
        
        # Translation quality estimation
        translation_quality = self._estimate_translation_quality(chains)
        
        return {
            'language_distribution_score': lang_distribution,
            'cross_lingual_semantic_consistency': semantic_consistency,
            'complexity_handling_score': complexity_handling,
            'cultural_adaptation_score': cultural_adaptation,
            'translation_quality_score': translation_quality,
            'advanced_multilingual_score': np.mean([
                lang_distribution, semantic_consistency, complexity_handling,
                cultural_adaptation, translation_quality
            ])
        }
    
    def _analyze_language_distribution(self, chains: List[Dict[str, Any]]) -> float:
        """Analyze distribution of languages in chains"""
        languages = [chain.get('language', 'en') for chain in chains]
        lang_counts = defaultdict(int)
        
        for lang in languages:
            lang_counts[lang] += 1
        
        # Compute distribution entropy (higher = more balanced)
        total = len(languages)
        entropy = -sum((count/total) * np.log2(count/total) for count in lang_counts.values())
        max_entropy = np.log2(len(lang_counts))
        
        return entropy / max_entropy if max_entropy > 0 else 0.0
    
    def _compute_cross_lingual_semantics(self, chains: List[Dict[str, Any]]) -> float:
        """Compute semantic consistency across languages"""
        lang_groups = defaultdict(list)
        
        for chain in chains:
            lang = chain.get('language', 'en')
            text = chain.get('text', '')
            # Simple semantic features: word count, sentence count
            features = [len(text.split()), text.count('.') + text.count('!') + text.count('?')]
            lang_groups[lang].append(features)
        
        if len(lang_groups) < 2:
            return 1.0
        
        # Compute feature consistency across languages
        lang_means = {}
        for lang, features_list in lang_groups.items():
            if features_list:
                lang_means[lang] = np.mean(features_list, axis=0)
        
        # Compute pairwise consistency
        consistencies = []
        languages = list(lang_means.keys())
        
        for i in range(len(languages)):
            for j in range(i + 1, len(languages)):
                lang1, lang2 = languages[i], languages[j]
                # Cosine similarity between feature vectors
                vec1, vec2 = lang_means[lang1], lang_means[lang2]
                similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                consistencies.append(max(0.0, similarity))
        
        return np.mean(consistencies) if consistencies else 0.0
    
    def _evaluate_complexity_handling(self, chains: List[Dict[str, Any]]) -> float:
        """Evaluate how well the system handles language complexity"""
        complexity_scores = []
        
        for chain in chains:
            lang = chain.get('language', 'en')
            text = chain.get('text', '')
            complexity_weight = self.language_complexity.get(lang, 1.0)
            
            # Quality indicators adjusted for language complexity
            text_quality = len(text) / max(complexity_weight * 100, 1)  # Expected length adjustment
            complexity_scores.append(min(text_quality, 1.0))
        
        return np.mean(complexity_scores) if complexity_scores else 0.0
    
    def _assess_cultural_adaptation(self, chains: List[Dict[str, Any]]) -> float:
        """Assess cultural adaptation in multilingual chains"""
        # Simple cultural adaptation based on language-specific patterns
        cultural_indicators = {
            'en': ['individual', 'efficiency', 'innovation'],
            'id': ['community', 'harmony', 'respect'],
            'zh': ['balance', 'tradition', 'collective'],
            'es': ['family', 'relationship', 'warmth'],
            'ar': ['honor', 'tradition', 'community']
        }
        
        adaptation_scores = []
        
        for chain in chains:
            lang = chain.get('language', 'en')
            text = chain.get('text', '').lower()
            indicators = cultural_indicators.get(lang, [])
            
            if indicators:
                matches = sum(1 for indicator in indicators if indicator in text)
                adaptation_score = matches / len(indicators)
                adaptation_scores.append(adaptation_score)
        
        return np.mean(adaptation_scores) if adaptation_scores else 0.0
    
    def _estimate_translation_quality(self, chains: List[Dict[str, Any]]) -> float:
        """Estimate translation quality based on text characteristics"""
        quality_scores = []
        
        for chain in chains:
            text = chain.get('text', '')
            
            # Quality indicators
            word_count = len(text.split())
            sentence_count = max(1, text.count('.') + text.count('!') + text.count('?'))
            avg_word_length = np.mean([len(word) for word in text.split()]) if text.split() else 0
            
            # Normalize quality score
            quality = min(1.0, (word_count / sentence_count) * (avg_word_length / 10))
            quality_scores.append(quality)
        
        return np.mean(quality_scores) if quality_scores else 0.0
    
    def _empty_multilingual_metrics(self) -> Dict[str, float]:
        """Return empty multilingual metrics"""
        return {
            'language_distribution_score': 0.0,
            'cross_lingual_semantic_consistency': 0.0,
            'complexity_handling_score': 0.0,
            'cultural_adaptation_score': 0.0,
            'translation_quality_score': 0.0,
            'advanced_multilingual_score': 0.0
        }

class EnhancedGraphConsistencyBenchmark:
    """Enhanced graph consistency benchmark with semantic analysis"""
    
    def __init__(self):
        self.concept_weights = {
            'analysis': 1.2,
            'reasoning': 1.3,
            'logic': 1.1,
            'creativity': 1.0,
            'system': 1.1
        }
    
    def compute_advanced_graph_metrics(self, chains: List[Dict[str, Any]], 
                                     graph_vocab: Dict[str, List[str]]) -> Dict[str, float]:
        """Compute advanced graph consistency metrics"""
        if not chains:
            return self._empty_graph_metrics()
        
        # Semantic density analysis
        semantic_density = self._compute_semantic_density(chains, graph_vocab)
        
        # Concept relationship mapping
        relationship_quality = self._analyze_concept_relationships(chains, graph_vocab)
        
        # Graph coverage completeness
        coverage_completeness = self._evaluate_coverage_completeness(chains, graph_vocab)
        
        # Semantic coherence in graph space
        graph_coherence = self._compute_graph_coherence(chains, graph_vocab)
        
        # Weighted concept importance
        concept_importance = self._evaluate_concept_importance(chains, graph_vocab)
        
        return {
            'semantic_density': semantic_density,
            'relationship_quality': relationship_quality,
            'coverage_completeness': coverage_completeness,
            'graph_coherence': graph_coherence,
            'concept_importance_score': concept_importance,
            'advanced_graph_score': np.mean([
                semantic_density, relationship_quality, coverage_completeness,
                graph_coherence, concept_importance
            ])
        }
    
    def _compute_semantic_density(self, chains: List[Dict[str, Any]], 
                                graph_vocab: Dict[str, List[str]]) -> float:
        """Compute semantic density of extracted concepts"""
        total_concepts = sum(len(category) for category in graph_vocab.values())
        total_extractions = 0
        unique_extractions = set()
        
        for chain in chains:
            nodes = chain.get('graph_nodes', [])
            total_extractions += len(nodes)
            unique_extractions.update(nodes)
        
        if total_extractions == 0:
            return 0.0
        
        # Density = unique concepts / total possible * extraction efficiency
        uniqueness_ratio = len(unique_extractions) / max(total_concepts, 1)
        extraction_efficiency = total_extractions / (len(chains) * 10)  # Expected ~10 concepts per chain
        
        return min(1.0, uniqueness_ratio * extraction_efficiency)
    
    def _analyze_concept_relationships(self, chains: List[Dict[str, Any]], 
                                     graph_vocab: Dict[str, List[str]]) -> float:
        """Analyze quality of concept relationships"""
        relations = graph_vocab.get('relations', [])
        concepts = graph_vocab.get('concepts', []) + graph_vocab.get('entities', [])
        
        relationship_scores = []
        
        for chain in chains:
            text = chain.get('text', '').lower()
            found_relations = [r for r in relations if r.lower() in text]
            found_concepts = [c for c in concepts if c.lower() in text]
            
            if found_concepts and found_relations:
                # Quality based on relation-to-concept ratio
                quality = len(found_relations) / len(found_concepts)
                relationship_scores.append(min(1.0, quality))
            else:
                relationship_scores.append(0.0)
        
        return np.mean(relationship_scores) if relationship_scores else 0.0
    
    def _evaluate_coverage_completeness(self, chains: List[Dict[str, Any]], 
                                      graph_vocab: Dict[str, List[str]]) -> float:
        """Evaluate completeness of graph vocabulary coverage"""
        all_vocab = []
        for category in graph_vocab.values():
            all_vocab.extend(category)
        
        covered_vocab = set()
        for chain in chains:
            nodes = chain.get('graph_nodes', [])
            covered_vocab.update(nodes)
        
        return len(covered_vocab) / max(len(all_vocab), 1)
    
    def _compute_graph_coherence(self, chains: List[Dict[str, Any]], 
                               graph_vocab: Dict[str, List[str]]) -> float:
        """Compute coherence of graph concepts across chains"""
        concept_co_occurrences = defaultdict(int)
        total_pairs = 0
        
        for chain in chains:
            nodes = chain.get('graph_nodes', [])
            # Count co-occurrences of concept pairs
            for i, node1 in enumerate(nodes):
                for node2 in nodes[i+1:]:
                    pair = tuple(sorted([node1, node2]))
                    concept_co_occurrences[pair] += 1
                    total_pairs += 1
        
        if total_pairs == 0:
            return 0.0
        
        # Coherence based on repeated concept relationships
        repeated_pairs = sum(1 for count in concept_co_occurrences.values() if count > 1)
        return repeated_pairs / max(len(concept_co_occurrences), 1)
    
    def _evaluate_concept_importance(self, chains: List[Dict[str, Any]], 
                                   graph_vocab: Dict[str, List[str]]) -> float:
        """Evaluate extraction of important concepts"""
        concept_counts = defaultdict(int)
        
        for chain in chains:
            nodes = chain.get('graph_nodes', [])
            for node in nodes:
                concept_counts[node] += 1
        
        # Weight by concept importance
        weighted_score = 0.0
        total_weight = 0.0
        
        for concept, count in concept_counts.items():
            weight = self.concept_weights.get(concept, 1.0)
            weighted_score += count * weight
            total_weight += weight
        
        return weighted_score / max(total_weight, 1)
    
    def _empty_graph_metrics(self) -> Dict[str, float]:
        """Return empty graph metrics"""
        return {
            'semantic_density': 0.0,
            'relationship_quality': 0.0,
            'coverage_completeness': 0.0,
            'graph_coherence': 0.0,
            'concept_importance_score': 0.0,
            'advanced_graph_score': 0.0
        }

class EnhancedPerformanceBenchmark:
    """Enhanced performance benchmark with detailed profiling"""
    
    def __init__(self):
        self.cpu_count = mp.cpu_count()
    
    def compute_advanced_performance_metrics(self, aligner: DCoTAgentAligner, 
                                           test_prompts: List[str]) -> Dict[str, Any]:
        """Compute advanced performance metrics"""
        # Memory usage profiling
        memory_metrics = self._profile_memory_usage(aligner, test_prompts[0])
        
        # Throughput analysis
        throughput_metrics = self._analyze_throughput(aligner, test_prompts)
        
        # Scalability assessment
        scalability_metrics = self._assess_scalability(aligner, test_prompts[0])
        
        # Resource utilization
        resource_metrics = self._profile_resource_utilization(aligner, test_prompts[0])
        
        # Latency distribution
        latency_metrics = self._analyze_latency_distribution(aligner, test_prompts[0])
        
        return {
            'memory_metrics': memory_metrics,
            'throughput_metrics': throughput_metrics,
            'scalability_metrics': scalability_metrics,
            'resource_metrics': resource_metrics,
            'latency_metrics': latency_metrics,
            'advanced_performance_score': self._compute_composite_performance_score({
                'memory': memory_metrics,
                'throughput': throughput_metrics,
                'scalability': scalability_metrics,
                'resources': resource_metrics,
                'latency': latency_metrics
            })
        }
    
    def _profile_memory_usage(self, aligner: DCoTAgentAligner, prompt: str) -> Dict[str, float]:
        """Profile memory usage during benchmark execution"""
        import psutil
        import gc
        
        # Baseline memory
        gc.collect()
        baseline_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # Execute benchmark operations
        chains = aligner.generate_chains(prompt, n=10, diverse_styles=True)
        peak_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        aligned_chains = aligner.align_to_graph(chains, aligner.graph_vocab)
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        return {
            'baseline_memory_mb': baseline_memory,
            'peak_memory_mb': peak_memory,
            'final_memory_mb': final_memory,
            'memory_increase_mb': peak_memory - baseline_memory,
            'memory_efficiency': 1.0 / max(peak_memory - baseline_memory, 1.0)
        }
    
    def _analyze_throughput(self, aligner: DCoTAgentAligner, prompts: List[str]) -> Dict[str, float]:
        """Analyze throughput across different operations"""
        # Chain generation throughput
        start_time = time.time()
        total_chains = 0
        
        for prompt in prompts[:3]:  # Use first 3 prompts
            chains = aligner.generate_chains(prompt, n=5, diverse_styles=True)
            total_chains += len(chains)
        
        generation_time = time.time() - start_time
        generation_throughput = total_chains / max(generation_time, 0.001)
        
        # Alignment throughput
        test_chains = aligner.generate_chains(prompts[0], n=20, diverse_styles=True)
        start_time = time.time()
        aligned_chains = aligner.align_to_graph(test_chains, aligner.graph_vocab)
        alignment_time = time.time() - start_time
        alignment_throughput = len(aligned_chains) / max(alignment_time, 0.001)
        
        return {
            'generation_throughput': generation_throughput,
            'alignment_throughput': alignment_throughput,
            'overall_throughput': (generation_throughput + alignment_throughput) / 2,
            'generation_time': generation_time,
            'alignment_time': alignment_time
        }
    
    def _assess_scalability(self, aligner: DCoTAgentAligner, prompt: str) -> Dict[str, float]:
        """Assess scalability characteristics"""
        scales = [1, 2, 5, 10]
        execution_times = []
        throughputs = []
        
        for scale in scales:
            start_time = time.time()
            chains = aligner.generate_chains(prompt, n=3 * scale, diverse_styles=True)
            execution_time = time.time() - start_time
            
            execution_times.append(execution_time)
            throughputs.append(len(chains) / max(execution_time, 0.001))
        
        # Compute scalability metrics
        time_complexity = np.polyfit(scales, execution_times, 1)[0]  # Linear coefficient
        throughput_stability = 1.0 - (np.std(throughputs) / np.mean(throughputs))
        
        return {
            'time_complexity_slope': time_complexity,
            'throughput_stability': max(0.0, throughput_stability),
            'scalability_score': max(0.0, 1.0 - time_complexity / max(scales)),
            'execution_times': execution_times,
            'throughputs': throughputs
        }
    
    def _profile_resource_utilization(self, aligner: DCoTAgentAligner, prompt: str) -> Dict[str, float]:
        """Profile CPU and resource utilization"""
        import psutil
        
        # Monitor resource usage during execution
        cpu_before = psutil.cpu_percent(interval=0.1)
        
        start_time = time.time()
        chains = aligner.generate_chains(prompt, n=15, diverse_styles=True)
        aligned_chains = aligner.align_to_graph(chains, aligner.graph_vocab)
        execution_time = time.time() - start_time
        
        cpu_after = psutil.cpu_percent(interval=0.1)
        
        return {
            'cpu_utilization_before': cpu_before,
            'cpu_utilization_after': cpu_after,
            'cpu_efficiency': min(1.0, len(aligned_chains) / max(cpu_after, 1.0)),
            'resource_efficiency': len(aligned_chains) / max(execution_time, 0.001),
            'execution_time': execution_time
        }
    
    def _analyze_latency_distribution(self, aligner: DCoTAgentAligner, prompt: str) -> Dict[str, float]:
        """Analyze latency distribution for individual operations"""
        latencies = []
        
        # Measure individual chain generation latencies
        for _ in range(10):
            start_time = time.time()
            chains = aligner.generate_chains(prompt, n=1, diverse_styles=True)
            latency = time.time() - start_time
            latencies.append(latency)
        
        return {
            'mean_latency': np.mean(latencies),
            'median_latency': np.median(latencies),
            'p95_latency': np.percentile(latencies, 95),
            'p99_latency': np.percentile(latencies, 99),
            'latency_std': np.std(latencies),
            'latency_consistency': 1.0 - (np.std(latencies) / max(np.mean(latencies), 0.001))
        }
    
    def _compute_composite_performance_score(self, metrics: Dict[str, Dict]) -> float:
        """Compute composite performance score"""
        # Extract key performance indicators
        memory_efficiency = metrics['memory'].get('memory_efficiency', 0.0)
        throughput_score = min(1.0, metrics['throughput'].get('overall_throughput', 0.0) / 1000)
        scalability_score = metrics['scalability'].get('scalability_score', 0.0)
        resource_efficiency = min(1.0, metrics['resources'].get('resource_efficiency', 0.0) / 100)
        latency_consistency = metrics['latency'].get('latency_consistency', 0.0)
        
        # Weighted composite score
        weights = [0.2, 0.25, 0.25, 0.15, 0.15]
        scores = [memory_efficiency, throughput_score, scalability_score, resource_efficiency, latency_consistency]
        
        return sum(w * s for w, s in zip(weights, scores))

class EnhancedBenchmarkSuite:
    """Enhanced benchmark suite with advanced analytics"""
    
    def __init__(self, parallel_execution: bool = True):
        self.parallel_execution = parallel_execution
        self.cognitive_benchmark = EnhancedCognitiveDiversityBenchmark()
        self.multilingual_benchmark = EnhancedMultilingualBenchmark()
        self.graph_benchmark = EnhancedGraphConsistencyBenchmark()
        self.performance_benchmark = EnhancedPerformanceBenchmark()
        self.statistical_analyzer = StatisticalAnalyzer()
        
        # Enhanced test configuration
        self.test_prompts = [
            "How can we develop more reliable and trustworthy artificial intelligence systems?",
            "What are the fundamental principles of effective problem-solving across different domains?",
            "Explain the relationship between creativity, systematic thinking, and innovation in research",
            "How do we balance technological advancement with ethical considerations and safety?",
            "What strategies promote effective cross-cultural communication and global collaboration?",
            "Describe the process of scientific discovery and knowledge validation in complex systems",
            "How can we address bias, fairness, and transparency in automated decision-making systems?"
        ]
        
        self.enhanced_graph_vocabulary = {
            'concepts': [
                'analysis', 'reasoning', 'logic', 'creativity', 'system', 'pattern',
                'evidence', 'hypothesis', 'theory', 'model', 'framework', 'method',
                'validation', 'evaluation', 'optimization', 'innovation', 'solution',
                'reliability', 'trustworthiness', 'ethics', 'fairness', 'transparency',
                'collaboration', 'communication', 'discovery', 'knowledge', 'complexity'
            ],
            'relations': [
                'causes', 'leads_to', 'requires', 'supports', 'contradicts',
                'enhances', 'influences', 'depends_on', 'enables', 'improves',
                'validates', 'explains', 'predicts', 'correlates_with', 'balances',
                'integrates', 'synthesizes', 'transforms', 'optimizes', 'facilitates'
            ],
            'entities': [
                'problem', 'solution', 'approach', 'strategy', 'outcome', 'result',
                'process', 'mechanism', 'component', 'factor', 'variable', 'parameter',
                'constraint', 'objective', 'criterion', 'metric', 'benchmark', 'system',
                'algorithm', 'model', 'framework', 'methodology', 'technique', 'tool'
            ]
        }
        
        self.languages = ['en', 'id', 'zh', 'es']
    
    def run_enhanced_benchmark(self, aligner: DCoTAgentAligner, 
                             config: AdvancedBenchmarkConfig) -> BenchmarkMetrics:
        """Run enhanced benchmark with statistical analysis"""
        print(f"Running enhanced benchmark: {config.name}")
        
        # Run multiple iterations for statistical significance
        iteration_results = []
        
        for iteration in range(config.test_iterations):
            print(f"  Iteration {iteration + 1}/{config.test_iterations}")
            
            # Single iteration benchmark
            single_result = self._run_single_iteration(aligner, config)
            iteration_results.append(single_result)
        
        # Statistical analysis
        composite_scores = [result['composite_score'] for result in iteration_results]
        confidence_interval = self.statistical_analyzer.compute_confidence_interval(
            composite_scores, config.confidence_level
        )
        
        # Compute mean metrics
        mean_metrics = self._compute_mean_metrics(iteration_results)
        
        return BenchmarkMetrics(
            cognitive_diversity=mean_metrics['cognitive_diversity'],
            multilingual_alignment=mean_metrics['multilingual_alignment'],
            graph_consistency=mean_metrics['graph_consistency'],
            performance_score=mean_metrics['performance_score'],
            composite_score=mean_metrics['composite_score'],
            execution_time=mean_metrics['execution_time'],
            confidence_interval=confidence_interval,
            statistical_significance=np.std(composite_scores)
        )
    
    def _run_single_iteration(self, aligner: DCoTAgentAligner, 
                            config: AdvancedBenchmarkConfig) -> Dict[str, float]:
        """Run single benchmark iteration"""
        start_time = time.time()
        
        # Generate test chains
        all_chains = []
        for prompt in self.test_prompts[:5]:  # Use first 5 prompts
            chains = aligner.generate_chains(prompt, n=4, diverse_styles=True)
            all_chains.extend(chains)
        
        # Generate multilingual chains
        multilingual_chains = []
        for prompt in self.test_prompts[:3]:
            ml_chains = aligner.generate_multilingual_chains(
                prompt, languages=config.languages, chains_per_lang=2
            )
            multilingual_chains.extend(ml_chains)
        
        # Align chains to graph
        aligned_chains = aligner.align_to_graph(all_chains, self.enhanced_graph_vocabulary)
        aligned_ml_chains = aligner.align_to_graph(multilingual_chains, self.enhanced_graph_vocabulary)
        
        # Run enhanced benchmarks
        cognitive_results = self.cognitive_benchmark.compute_advanced_diversity_metrics(aligned_chains)
        multilingual_results = self.multilingual_benchmark.compute_advanced_multilingual_metrics(aligned_ml_chains)
        graph_results = self.graph_benchmark.compute_advanced_graph_metrics(aligned_chains, self.enhanced_graph_vocabulary)
        performance_results = self.performance_benchmark.compute_advanced_performance_metrics(aligner, self.test_prompts[:2])
        
        execution_time = time.time() - start_time
        
        # Extract key scores
        cognitive_score = cognitive_results.get('advanced_diversity_score', 0.0)
        multilingual_score = multilingual_results.get('advanced_multilingual_score', 0.0)
        graph_score = graph_results.get('advanced_graph_score', 0.0)
        performance_score = performance_results.get('advanced_performance_score', 0.0)
        
        # Compute composite score
        weights = {'cognitive': 0.3, 'multilingual': 0.25, 'graph': 0.25, 'performance': 0.2}
        composite_score = (
            weights['cognitive'] * cognitive_score +
            weights['multilingual'] * multilingual_score +
            weights['graph'] * graph_score +
            weights['performance'] * performance_score
        )
        
        return {
            'cognitive_diversity': cognitive_score,
            'multilingual_alignment': multilingual_score,
            'graph_consistency': graph_score,
            'performance_score': performance_score,
            'composite_score': composite_score,
            'execution_time': execution_time,
            'detailed_results': {
                'cognitive': cognitive_results,
                'multilingual': multilingual_results,
                'graph': graph_results,
                'performance': performance_results
            }
        }
    
    def _compute_mean_metrics(self, iteration_results: List[Dict[str, float]]) -> Dict[str, float]:
        """Compute mean metrics across iterations"""
        return {
            'cognitive_diversity': np.mean([r['cognitive_diversity'] for r in iteration_results]),
            'multilingual_alignment': np.mean([r['multilingual_alignment'] for r in iteration_results]),
            'graph_consistency': np.mean([r['graph_consistency'] for r in iteration_results]),
            'performance_score': np.mean([r['performance_score'] for r in iteration_results]),
            'composite_score': np.mean([r['composite_score'] for r in iteration_results]),
            'execution_time': np.mean([r['execution_time'] for r in iteration_results])
        }
    
    def run_comparative_enhanced_benchmark(self, configurations: Dict[str, AdvancedBenchmarkConfig]) -> Dict[str, Any]:
        """Run comparative benchmark with enhanced analytics"""
        print("Running Enhanced Comparative Benchmark")
        print("=" * 45)
        
        results = {}
        mock_llm = MockLLM()
        
        # Run benchmark for each configuration
        for config_name, config in configurations.items():
            print(f"\nTesting configuration: {config_name}")
            
            # Create aligner
            aligner = DCoTAgentAligner(
                model=mock_llm,
                lang='en',
                graph_vocab=self.enhanced_graph_vocabulary
            )
            aligner.reasoning_styles = config.reasoning_styles
            
            # Run enhanced benchmark
            benchmark_result = self.run_enhanced_benchmark(aligner, config)
            results[config_name] = benchmark_result
        
        # Statistical comparison
        comparative_analysis = self._perform_statistical_comparison(results)
        
        return {
            'individual_results': results,
            'comparative_analysis': comparative_analysis,
            'statistical_summary': self._generate_statistical_summary(results)
        }
    
    def _perform_statistical_comparison(self, results: Dict[str, BenchmarkMetrics]) -> Dict[str, Any]:
        """Perform statistical comparison between configurations"""
        config_names = list(results.keys())
        comparison_matrix = {}
        
        # Pairwise comparisons
        for i, config1 in enumerate(config_names):
            for config2 in config_names[i+1:]:
                result1, result2 = results[config1], results[config2]
                
                # Effect size (simplified - using composite scores)
                effect_size = abs(result1.composite_score - result2.composite_score)
                
                # Statistical significance (simplified)
                significance = abs(result1.statistical_significance - result2.statistical_significance)
                
                comparison_matrix[f"{config1}_vs_{config2}"] = {
                    'effect_size': effect_size,
                    'significance': significance,
                    'better_config': config1 if result1.composite_score > result2.composite_score else config2,
                    'performance_difference': abs(result1.composite_score - result2.composite_score)
                }
        
        return comparison_matrix
    
    def _generate_statistical_summary(self, results: Dict[str, BenchmarkMetrics]) -> Dict[str, Any]:
        """Generate statistical summary of results"""
        composite_scores = [result.composite_score for result in results.values()]
        execution_times = [result.execution_time for result in results.values()]
        
        return {
            'composite_score_stats': {
                'mean': np.mean(composite_scores),
                'std': np.std(composite_scores),
                'min': np.min(composite_scores),
                'max': np.max(composite_scores),
                'range': np.max(composite_scores) - np.min(composite_scores)
            },
            'execution_time_stats': {
                'mean': np.mean(execution_times),
                'std': np.std(execution_times),
                'min': np.min(execution_times),
                'max': np.max(execution_times)
            },
            'best_configuration': max(results.keys(), key=lambda k: results[k].composite_score),
            'most_consistent': min(results.keys(), key=lambda k: results[k].statistical_significance)
        }

def main():
    """Run enhanced benchmarking demonstration"""
    print("Enhanced DCoTAgentAligner Benchmark Suite")
    print("=" * 50)
    
    # Create enhanced configurations
    configurations = {
        'standard_enhanced': AdvancedBenchmarkConfig(
            name='Standard Enhanced',
            reasoning_styles=['analytical', 'creative', 'systematic', 'intuitive', 'critical'],
            languages=['en', 'id', 'zh'],
            graph_vocab_size=75,
            test_iterations=3,
            statistical_analysis=True
        ),
        'performance_optimized': AdvancedBenchmarkConfig(
            name='Performance Optimized',
            reasoning_styles=['analytical', 'systematic'],
            languages=['en', 'id'],
            graph_vocab_size=50,
            test_iterations=3,
            statistical_analysis=True
        ),
        'diversity_maximized': AdvancedBenchmarkConfig(
            name='Diversity Maximized',
            reasoning_styles=['analytical', 'creative', 'systematic', 'intuitive', 'critical'],
            languages=['en', 'id', 'zh', 'es'],
            graph_vocab_size=75,
            test_iterations=3,
            statistical_analysis=True
        )
    }
    
    # Run enhanced comparative benchmark
    enhanced_suite = EnhancedBenchmarkSuite()
    results = enhanced_suite.run_comparative_enhanced_benchmark(configurations)
    
    # Display results
    print("\n" + "=" * 50)
    print("ENHANCED BENCHMARK RESULTS")
    print("=" * 50)
    
    for config_name, result in results['individual_results'].items():
        print(f"\n{config_name}:")
        print(f"  Composite Score: {result.composite_score:.3f} ± {result.statistical_significance:.3f}")
        print(f"  Confidence Interval: [{result.confidence_interval[0]:.3f}, {result.confidence_interval[1]:.3f}]")
        print(f"  Cognitive Diversity: {result.cognitive_diversity:.3f}")
        print(f"  Multilingual Alignment: {result.multilingual_alignment:.3f}")
        print(f"  Graph Consistency: {result.graph_consistency:.3f}")
        print(f"  Performance Score: {result.performance_score:.3f}")
        print(f"  Execution Time: {result.execution_time:.3f}s")
    
    # Statistical summary
    stats = results['statistical_summary']
    print(f"\nStatistical Summary:")
    print(f"  Best Configuration: {stats['best_configuration']}")
    print(f"  Most Consistent: {stats['most_consistent']}")
    print(f"  Score Range: {stats['composite_score_stats']['range']:.3f}")
    
    # Save enhanced results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Convert results to serializable format
    serializable_results = {}
    for config_name, result in results['individual_results'].items():
        serializable_results[config_name] = result.to_dict()
    
    enhanced_results = {
        'individual_results': serializable_results,
        'comparative_analysis': results['comparative_analysis'],
        'statistical_summary': results['statistical_summary'],
        'metadata': {
            'timestamp': timestamp,
            'benchmark_type': 'enhanced',
            'configurations_tested': len(configurations)
        }
    }
    
    with open(f'enhanced_benchmark_results_{timestamp}.json', 'w', encoding='utf-8') as f:
        json.dump(enhanced_results, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\nEnhanced results saved to: enhanced_benchmark_results_{timestamp}.json")
    print("Enhanced benchmarking completed successfully!")
    
    return enhanced_results

if __name__ == "__main__":
    results = main()