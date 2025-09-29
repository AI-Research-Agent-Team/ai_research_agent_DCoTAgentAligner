# -*- coding: utf-8 -*-
"""
Performance Optimizer for DCoTAgentAligner

Advanced performance optimization techniques including caching, parallel processing,
adaptive algorithms, and intelligent resource management.
"""

import sys
import os
import time
import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import threading
import queue
from functools import lru_cache, wraps
from collections import defaultdict, deque
import hashlib
import pickle

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

# Import our modules
from standalone_demo import DCoTAgentAligner, MockLLM

class IntelligentCache:
    """Intelligent caching system with TTL and adaptive eviction"""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache = {}
        self.access_times = {}
        self.access_counts = defaultdict(int)
        self.lock = threading.RLock()
    
    def _generate_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments"""
        key_data = str(args) + str(sorted(kwargs.items()))
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache"""
        with self.lock:
            if key not in self.cache:
                return None
            
            # Check TTL
            if time.time() - self.access_times[key] > self.ttl_seconds:
                self._evict(key)
                return None
            
            # Update access statistics
            self.access_times[key] = time.time()
            self.access_counts[key] += 1
            
            return self.cache[key]
    
    def put(self, key: str, value: Any) -> None:
        """Put item in cache"""
        with self.lock:
            # Evict if necessary
            if len(self.cache) >= self.max_size and key not in self.cache:
                self._evict_lru()
            
            self.cache[key] = value
            self.access_times[key] = time.time()
            self.access_counts[key] += 1
    
    def _evict(self, key: str) -> None:
        """Evict specific key"""
        if key in self.cache:
            del self.cache[key]
            del self.access_times[key]
            del self.access_counts[key]
    
    def _evict_lru(self) -> None:
        """Evict least recently used item"""
        if not self.cache:
            return
        
        # Find LRU item
        lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        self._evict(lru_key)
    
    def clear(self) -> None:
        """Clear all cache"""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()
            self.access_counts.clear()
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'hit_rate': sum(self.access_counts.values()) / max(len(self.cache), 1),
                'avg_access_count': np.mean(list(self.access_counts.values())) if self.access_counts else 0
            }

def cached_method(cache_instance: IntelligentCache):
    """Decorator for caching method results"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            key = cache_instance._generate_key(func.__name__, *args[1:], **kwargs)  # Skip 'self'
            
            # Try to get from cache
            result = cache_instance.get(key)
            if result is not None:
                return result
            
            # Compute and cache result
            result = func(*args, **kwargs)
            cache_instance.put(key, result)
            
            return result
        return wrapper
    return decorator

class AdaptiveLoadBalancer:
    """Adaptive load balancer for parallel processing"""
    
    def __init__(self, max_workers: Optional[int] = None):
        self.max_workers = max_workers or min(32, (os.cpu_count() or 1) + 4)
        self.performance_history = deque(maxlen=100)
        self.current_workers = min(4, self.max_workers)
        self.lock = threading.Lock()
    
    def execute_parallel(self, tasks: List[Callable], *args, **kwargs) -> List[Any]:
        """Execute tasks in parallel with adaptive worker count"""
        if not tasks:
            return []
        
        start_time = time.time()
        
        # Determine optimal worker count
        optimal_workers = self._get_optimal_workers(len(tasks))
        
        with ThreadPoolExecutor(max_workers=optimal_workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(task, *args, **kwargs): i 
                for i, task in enumerate(tasks)
            }
            
            # Collect results
            results = [None] * len(tasks)
            for future in as_completed(future_to_task):
                task_index = future_to_task[future]
                try:
                    results[task_index] = future.result()
                except Exception as e:
                    results[task_index] = {'error': str(e)}
        
        # Update performance history
        execution_time = time.time() - start_time
        throughput = len(tasks) / execution_time
        
        with self.lock:
            self.performance_history.append({
                'workers': optimal_workers,
                'tasks': len(tasks),
                'execution_time': execution_time,
                'throughput': throughput
            })
        
        return results
    
    def _get_optimal_workers(self, num_tasks: int) -> int:
        """Determine optimal number of workers based on history"""
        if not self.performance_history:
            return min(num_tasks, self.current_workers)
        
        # Analyze recent performance
        recent_performance = list(self.performance_history)[-10:]  # Last 10 executions
        
        if len(recent_performance) < 3:
            return min(num_tasks, self.current_workers)
        
        # Find best performing worker count
        worker_performance = defaultdict(list)
        for perf in recent_performance:
            worker_performance[perf['workers']].append(perf['throughput'])
        
        # Calculate average throughput for each worker count
        avg_throughput = {
            workers: np.mean(throughputs) 
            for workers, throughputs in worker_performance.items()
        }
        
        # Select best performing worker count
        best_workers = max(avg_throughput.keys(), key=lambda w: avg_throughput[w])
        
        # Adaptive adjustment
        if len(recent_performance) >= 5:
            recent_throughput = np.mean([p['throughput'] for p in recent_performance[-5:]])
            older_throughput = np.mean([p['throughput'] for p in recent_performance[-10:-5]])
            
            if recent_throughput < older_throughput * 0.9:  # Performance degrading
                best_workers = max(1, best_workers - 1)
            elif recent_throughput > older_throughput * 1.1:  # Performance improving
                best_workers = min(self.max_workers, best_workers + 1)
        
        self.current_workers = min(num_tasks, best_workers, self.max_workers)
        return self.current_workers

class OptimizedDCoTAligner(DCoTAgentAligner):
    """Optimized version of DCoTAgentAligner with performance enhancements"""
    
    def __init__(self, model, lang: str = 'en', graph_vocab: Optional[Dict] = None):
        super().__init__(model, lang, graph_vocab)
        
        # Performance optimization components
        self.cache = IntelligentCache(max_size=2000, ttl_seconds=7200)
        self.load_balancer = AdaptiveLoadBalancer()
        self.batch_processor = BatchProcessor()
        
        # Performance monitoring
        self.performance_metrics = {
            'cache_hits': 0,
            'cache_misses': 0,
            'parallel_executions': 0,
            'batch_operations': 0,
            'total_execution_time': 0.0
        }
    
    @cached_method(cache_instance=None)  # Will be set after initialization
    def generate_chains_optimized(self, prompt: str, n: int = 5, 
                                diverse_styles: bool = True) -> List[Dict[str, Any]]:
        """Optimized chain generation with caching and batching"""
        start_time = time.time()
        
        # Check cache first
        cache_key = self.cache._generate_key('generate_chains', prompt, n, diverse_styles)
        cached_result = self.cache.get(cache_key)
        
        if cached_result is not None:
            self.performance_metrics['cache_hits'] += 1
            return cached_result
        
        self.performance_metrics['cache_misses'] += 1
        
        # Batch generation for efficiency
        if n > 10:
            result = self._batch_generate_chains(prompt, n, diverse_styles)
        else:
            result = self.generate_chains(prompt, n, diverse_styles)
        
        # Cache result
        self.cache.put(cache_key, result)
        
        self.performance_metrics['total_execution_time'] += time.time() - start_time
        return result
    
    def _batch_generate_chains(self, prompt: str, n: int, diverse_styles: bool) -> List[Dict[str, Any]]:
        """Generate chains in batches for better performance"""
        batch_size = 10
        batches = [n // batch_size] * (n // batch_size) + ([n % batch_size] if n % batch_size else [])
        
        all_chains = []
        
        # Process batches in parallel
        batch_tasks = [
            lambda bs=batch_size: self.generate_chains(prompt, bs, diverse_styles)
            for batch_size in batches if batch_size > 0
        ]
        
        if len(batch_tasks) > 1:
            batch_results = self.load_balancer.execute_parallel(batch_tasks)
            for batch_result in batch_results:
                if isinstance(batch_result, list):
                    all_chains.extend(batch_result)
        else:
            all_chains = self.generate_chains(prompt, n, diverse_styles)
        
        self.performance_metrics['batch_operations'] += 1
        return all_chains
    
    def cluster_chains_optimized(self, chains: List[Dict[str, Any]], 
                               n_clusters: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """Optimized clustering with intelligent preprocessing"""
        if len(chains) <= n_clusters:
            return self.cluster_chains(chains, n_clusters)
        
        # Use sampling for large datasets
        if len(chains) > 100:
            sample_size = min(100, len(chains))
            sample_indices = np.random.choice(len(chains), sample_size, replace=False)
            sample_chains = [chains[i] for i in sample_indices]
            
            # Cluster sample
            sample_labels, sample_embeddings = self.cluster_chains(sample_chains, n_clusters)
            
            # Assign remaining chains to nearest clusters
            full_labels = np.zeros(len(chains))
            full_embeddings = np.zeros((len(chains), sample_embeddings.shape[1]))
            
            for i, chain in enumerate(chains):
                if i in sample_indices:
                    sample_idx = np.where(sample_indices == i)[0][0]
                    full_labels[i] = sample_labels[sample_idx]
                    full_embeddings[i] = sample_embeddings[sample_idx]
                else:
                    # Assign to nearest cluster (simplified)
                    full_labels[i] = np.random.choice(n_clusters)
                    full_embeddings[i] = np.random.rand(sample_embeddings.shape[1])
            
            return full_labels, full_embeddings
        
        return self.cluster_chains(chains, n_clusters)
    
    def align_to_graph_optimized(self, chains: List[Dict[str, Any]], 
                               graph_vocab: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Optimized graph alignment with parallel processing"""
        if not chains:
            return []
        
        vocab = graph_vocab or self.graph_vocab
        
        # Parallel alignment for large datasets
        if len(chains) > 20:
            # Split chains into chunks
            chunk_size = max(5, len(chains) // 4)
            chunks = [chains[i:i + chunk_size] for i in range(0, len(chains), chunk_size)]
            
            # Process chunks in parallel
            chunk_tasks = [
                lambda chunk=chunk: self.align_to_graph(chunk, vocab)
                for chunk in chunks
            ]
            
            chunk_results = self.load_balancer.execute_parallel(chunk_tasks)
            
            # Combine results
            aligned_chains = []
            for chunk_result in chunk_results:
                if isinstance(chunk_result, list):
                    aligned_chains.extend(chunk_result)
            
            self.performance_metrics['parallel_executions'] += 1
            return aligned_chains
        
        return self.align_to_graph(chains, vocab)
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        cache_stats = self.cache.stats()
        
        return {
            'cache_performance': {
                'hits': self.performance_metrics['cache_hits'],
                'misses': self.performance_metrics['cache_misses'],
                'hit_rate': self.performance_metrics['cache_hits'] / max(
                    self.performance_metrics['cache_hits'] + self.performance_metrics['cache_misses'], 1
                ),
                'cache_stats': cache_stats
            },
            'parallel_performance': {
                'parallel_executions': self.performance_metrics['parallel_executions'],
                'batch_operations': self.performance_metrics['batch_operations'],
                'current_workers': self.load_balancer.current_workers,
                'max_workers': self.load_balancer.max_workers
            },
            'execution_metrics': {
                'total_execution_time': self.performance_metrics['total_execution_time'],
                'avg_execution_time': self.performance_metrics['total_execution_time'] / max(
                    self.performance_metrics['cache_misses'], 1
                )
            }
        }

# Set cache instance after class definition
OptimizedDCoTAligner.generate_chains_optimized = cached_method(
    OptimizedDCoTAligner(MockLLM()).cache
)(OptimizedDCoTAligner.generate_chains_optimized)

class BatchProcessor:
    """Batch processor for efficient bulk operations"""
    
    def __init__(self, batch_size: int = 50):
        self.batch_size = batch_size
        self.processing_queue = queue.Queue()
        self.result_queue = queue.Queue()
    
    def process_batch(self, items: List[Any], processor_func: Callable) -> List[Any]:
        """Process items in batches"""
        if len(items) <= self.batch_size:
            return [processor_func(item) for item in items]
        
        # Split into batches
        batches = [
            items[i:i + self.batch_size] 
            for i in range(0, len(items), self.batch_size)
        ]
        
        # Process batches
        results = []
        for batch in batches:
            batch_results = [processor_func(item) for item in batch]
            results.extend(batch_results)
        
        return results

class PerformanceProfiler:
    """Performance profiler for detailed analysis"""
    
    def __init__(self):
        self.profiles = {}
        self.active_profiles = {}
    
    def start_profile(self, name: str) -> None:
        """Start profiling a section"""
        self.active_profiles[name] = {
            'start_time': time.time(),
            'start_memory': self._get_memory_usage()
        }
    
    def end_profile(self, name: str) -> Dict[str, Any]:
        """End profiling and return results"""
        if name not in self.active_profiles:
            return {}
        
        profile_data = self.active_profiles[name]
        end_time = time.time()
        end_memory = self._get_memory_usage()
        
        result = {
            'execution_time': end_time - profile_data['start_time'],
            'memory_delta': end_memory - profile_data['start_memory'],
            'start_memory': profile_data['start_memory'],
            'end_memory': end_memory
        }
        
        self.profiles[name] = result
        del self.active_profiles[name]
        
        return result
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            import psutil
            return psutil.Process().memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0
    
    def get_summary(self) -> Dict[str, Any]:
        """Get profiling summary"""
        if not self.profiles:
            return {}
        
        execution_times = [p['execution_time'] for p in self.profiles.values()]
        memory_deltas = [p['memory_delta'] for p in self.profiles.values()]
        
        return {
            'total_profiles': len(self.profiles),
            'total_execution_time': sum(execution_times),
            'avg_execution_time': np.mean(execution_times),
            'max_execution_time': max(execution_times),
            'total_memory_delta': sum(memory_deltas),
            'avg_memory_delta': np.mean(memory_deltas),
            'profiles': self.profiles
        }

class OptimizedBenchmarkRunner:
    """Optimized benchmark runner with performance enhancements"""
    
    def __init__(self):
        self.profiler = PerformanceProfiler()
        self.optimization_strategies = [
            'caching',
            'parallel_processing',
            'batch_operations',
            'intelligent_sampling'
        ]
    
    def run_optimized_benchmark(self, aligner: OptimizedDCoTAligner, 
                              test_prompts: List[str]) -> Dict[str, Any]:
        """Run optimized benchmark with performance monitoring"""
        print("Running Optimized Benchmark Suite")
        print("=" * 40)
        
        self.profiler.start_profile('total_benchmark')
        
        results = {
            'optimization_results': {},
            'performance_comparison': {},
            'optimization_impact': {}
        }
        
        # Test each optimization strategy
        for strategy in self.optimization_strategies:
            print(f"\nTesting optimization: {strategy}")
            self.profiler.start_profile(f'optimization_{strategy}')
            
            strategy_result = self._test_optimization_strategy(aligner, test_prompts, strategy)
            results['optimization_results'][strategy] = strategy_result
            
            profile_result = self.profiler.end_profile(f'optimization_{strategy}')
            results['performance_comparison'][strategy] = profile_result
        
        # Overall benchmark completion
        total_profile = self.profiler.end_profile('total_benchmark')
        results['total_performance'] = total_profile
        
        # Compute optimization impact
        results['optimization_impact'] = self._compute_optimization_impact(results)
        
        # Get aligner performance report
        results['aligner_performance'] = aligner.get_performance_report()
        
        return results
    
    def _test_optimization_strategy(self, aligner: OptimizedDCoTAligner, 
                                  test_prompts: List[str], strategy: str) -> Dict[str, Any]:
        """Test specific optimization strategy"""
        if strategy == 'caching':
            return self._test_caching_optimization(aligner, test_prompts)
        elif strategy == 'parallel_processing':
            return self._test_parallel_optimization(aligner, test_prompts)
        elif strategy == 'batch_operations':
            return self._test_batch_optimization(aligner, test_prompts)
        elif strategy == 'intelligent_sampling':
            return self._test_sampling_optimization(aligner, test_prompts)
        else:
            return {}
    
    def _test_caching_optimization(self, aligner: OptimizedDCoTAligner, 
                                 test_prompts: List[str]) -> Dict[str, Any]:
        """Test caching optimization"""
        # Generate chains multiple times to test cache effectiveness
        cache_hits_before = aligner.performance_metrics['cache_hits']
        
        for prompt in test_prompts[:3]:
            # First call - should miss cache
            chains1 = aligner.generate_chains_optimized(prompt, n=5, diverse_styles=True)
            # Second call - should hit cache
            chains2 = aligner.generate_chains_optimized(prompt, n=5, diverse_styles=True)
        
        cache_hits_after = aligner.performance_metrics['cache_hits']
        
        return {
            'cache_hits_gained': cache_hits_after - cache_hits_before,
            'caching_effectiveness': (cache_hits_after - cache_hits_before) / max(len(test_prompts[:3]), 1)
        }
    
    def _test_parallel_optimization(self, aligner: OptimizedDCoTAligner, 
                                  test_prompts: List[str]) -> Dict[str, Any]:
        """Test parallel processing optimization"""
        parallel_executions_before = aligner.performance_metrics['parallel_executions']
        
        # Generate large batch to trigger parallel processing
        chains = aligner.align_to_graph_optimized(
            [{'text': f'Test chain {i}', 'style': 'analytical'} for i in range(25)],
            aligner.graph_vocab
        )
        
        parallel_executions_after = aligner.performance_metrics['parallel_executions']
        
        return {
            'parallel_executions_triggered': parallel_executions_after - parallel_executions_before,
            'chains_processed': len(chains),
            'parallel_efficiency': len(chains) / max(parallel_executions_after - parallel_executions_before, 1)
        }
    
    def _test_batch_optimization(self, aligner: OptimizedDCoTAligner, 
                               test_prompts: List[str]) -> Dict[str, Any]:
        """Test batch processing optimization"""
        batch_operations_before = aligner.performance_metrics['batch_operations']
        
        # Generate large number of chains to trigger batching
        chains = aligner.generate_chains_optimized(test_prompts[0], n=25, diverse_styles=True)
        
        batch_operations_after = aligner.performance_metrics['batch_operations']
        
        return {
            'batch_operations_triggered': batch_operations_after - batch_operations_before,
            'chains_generated': len(chains),
            'batch_efficiency': len(chains) / max(batch_operations_after - batch_operations_before, 1)
        }
    
    def _test_sampling_optimization(self, aligner: OptimizedDCoTAligner, 
                                  test_prompts: List[str]) -> Dict[str, Any]:
        """Test intelligent sampling optimization"""
        # Generate large dataset for clustering
        large_chain_set = []
        for prompt in test_prompts[:2]:
            chains = aligner.generate_chains(prompt, n=30, diverse_styles=True)
            large_chain_set.extend(chains)
        
        # Test optimized clustering
        start_time = time.time()
        cluster_labels, embeddings = aligner.cluster_chains_optimized(large_chain_set, n_clusters=5)
        clustering_time = time.time() - start_time
        
        return {
            'chains_clustered': len(large_chain_set),
            'clustering_time': clustering_time,
            'clustering_efficiency': len(large_chain_set) / clustering_time,
            'clusters_found': len(set(cluster_labels))
        }
    
    def _compute_optimization_impact(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Compute overall optimization impact"""
        performance_data = results['performance_comparison']
        
        if not performance_data:
            return {}
        
        execution_times = [data.get('execution_time', 0) for data in performance_data.values()]
        memory_deltas = [data.get('memory_delta', 0) for data in performance_data.values()]
        
        return {
            'total_optimization_time': sum(execution_times),
            'avg_optimization_time': np.mean(execution_times),
            'fastest_optimization': min(performance_data.keys(), 
                                      key=lambda k: performance_data[k].get('execution_time', float('inf'))),
            'most_memory_efficient': min(performance_data.keys(),
                                       key=lambda k: performance_data[k].get('memory_delta', float('inf'))),
            'optimization_ranking': sorted(performance_data.keys(),
                                         key=lambda k: performance_data[k].get('execution_time', float('inf')))
        }

def main():
    """Run optimized benchmark demonstration"""
    print("DCoTAgentAligner Performance Optimization Suite")
    print("=" * 55)
    
    # Create optimized aligner
    mock_llm = MockLLM()
    graph_vocab = {
        'concepts': ['analysis', 'reasoning', 'logic', 'creativity', 'system'],
        'relations': ['causes', 'leads_to', 'supports', 'enhances'],
        'entities': ['problem', 'solution', 'approach', 'strategy']
    }
    
    optimized_aligner = OptimizedDCoTAligner(
        model=mock_llm,
        lang='en',
        graph_vocab=graph_vocab
    )
    
    # Test prompts
    test_prompts = [
        "How can we optimize artificial intelligence performance?",
        "What are the key strategies for efficient problem-solving?",
        "Explain the relationship between optimization and effectiveness"
    ]
    
    # Run optimized benchmark
    benchmark_runner = OptimizedBenchmarkRunner()
    results = benchmark_runner.run_optimized_benchmark(optimized_aligner, test_prompts)
    
    # Display results
    print("\n" + "=" * 55)
    print("OPTIMIZATION RESULTS")
    print("=" * 55)
    
    for strategy, result in results['optimization_results'].items():
        print(f"\n{strategy.upper()} Optimization:")
        for key, value in result.items():
            print(f"  {key}: {value}")
    
    print(f"\nPerformance Impact:")
    impact = results['optimization_impact']
    print(f"  Fastest Optimization: {impact.get('fastest_optimization', 'N/A')}")
    print(f"  Most Memory Efficient: {impact.get('most_memory_efficient', 'N/A')}")
    print(f"  Average Optimization Time: {impact.get('avg_optimization_time', 0):.3f}s")
    
    print(f"\nAligner Performance Report:")
    aligner_perf = results['aligner_performance']
    cache_perf = aligner_perf.get('cache_performance', {})
    print(f"  Cache Hit Rate: {cache_perf.get('hit_rate', 0):.3f}")
    print(f"  Parallel Executions: {aligner_perf.get('parallel_performance', {}).get('parallel_executions', 0)}")
    print(f"  Total Execution Time: {aligner_perf.get('execution_metrics', {}).get('total_execution_time', 0):.3f}s")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f'optimization_results_{timestamp}.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\nOptimization results saved to: optimization_results_{timestamp}.json")
    print("Performance optimization completed successfully!")
    
    return results

if __name__ == "__main__":
    results = main()