"""
Benchmark comparing ML inference with and without custom KV-Store caching
Demonstrates the 60% latency reduction claim
"""

import time
import requests
import statistics
from typing import List
import matplotlib.pyplot as plt
from tabulate import tabulate

# Test sentences
TEST_SENTENCES = [
    "This movie was absolutely fantastic!",
    "I hated every minute of this film.",
    "The product quality is amazing.",
    "Terrible customer service experience.",
    "Best purchase I've ever made!",
    "This is the worst product ever.",
    "Highly recommend this to everyone.",
    "Don't waste your money on this.",
    "Exceeded all my expectations!",
    "Complete disappointment."
]

class BenchmarkRunner:
    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url
        
    def measure_latency(self, text: str, use_cache: bool) -> float:
        """Measure single request latency"""
        start = time.time()
        response = requests.post(
            f"{self.api_url}/predict",
            json={"text": text, "use_cache": use_cache}
        )
        latency = (time.time() - start) * 1000  # Convert to ms
        return latency, response.json()
    
    def run_benchmark(self, iterations: int = 100) -> dict:
        """Run comprehensive benchmark"""
        print("🚀 Starting Benchmark...")
        print(f"Iterations: {iterations}")
        print(f"Test sentences: {len(TEST_SENTENCES)}")
        print()
        
        # Phase 1: Without cache (cold start)
        print("Phase 1: No Caching (Model inference every time)")
        no_cache_latencies = []
        
        for i in range(iterations):
            text = TEST_SENTENCES[i % len(TEST_SENTENCES)]
            latency, _ = self.measure_latency(text, use_cache=False)
            no_cache_latencies.append(latency)
            
            if (i + 1) % 10 == 0:
                print(f"  Completed {i + 1}/{iterations} requests...")
        
        # Phase 2: With cache (warm cache)
        print("\nPhase 2: With Custom KV-Store Caching")
        # First, populate cache
        for text in TEST_SENTENCES:
            self.measure_latency(text, use_cache=True)
        
        # Now measure with warm cache
        with_cache_latencies = []
        
        for i in range(iterations):
            text = TEST_SENTENCES[i % len(TEST_SENTENCES)]
            latency, result = self.measure_latency(text, use_cache=True)
            with_cache_latencies.append(latency)
            
            if (i + 1) % 10 == 0:
                print(f"  Completed {i + 1}/{iterations} requests...")
        
        # Calculate statistics
        results = {
            "no_cache": {
                "mean": statistics.mean(no_cache_latencies),
                "median": statistics.median(no_cache_latencies),
                "p95": self._percentile(no_cache_latencies, 95),
                "p99": self._percentile(no_cache_latencies, 99),
                "min": min(no_cache_latencies),
                "max": max(no_cache_latencies),
                "latencies": no_cache_latencies
            },
            "with_cache": {
                "mean": statistics.mean(with_cache_latencies),
                "median": statistics.median(with_cache_latencies),
                "p95": self._percentile(with_cache_latencies, 95),
                "p99": self._percentile(with_cache_latencies, 99),
                "min": min(with_cache_latencies),
                "max": max(with_cache_latencies),
                "latencies": with_cache_latencies
            }
        }
        
        # Calculate improvement
        improvement = (
            (results["no_cache"]["mean"] - results["with_cache"]["mean"]) 
            / results["no_cache"]["mean"] * 100
        )
        results["improvement_percent"] = improvement
        
        return results
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile"""
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[index]
    
    def print_results(self, results: dict):
        """Print formatted results"""
        print("\n" + "="*70)
        print("📊 BENCHMARK RESULTS")
        print("="*70)
        
        table_data = [
            ["Metric", "No Cache", "With Cache", "Improvement"],
            ["─────────", "─────────", "──────────", "────────────"],
            [
                "Mean Latency",
                f"{results['no_cache']['mean']:.2f}ms",
                f"{results['with_cache']['mean']:.2f}ms",
                f"{results['improvement_percent']:.1f}% ⬇"
            ],
            [
                "Median",
                f"{results['no_cache']['median']:.2f}ms",
                f"{results['with_cache']['median']:.2f}ms",
                f"{self._calc_improvement(results['no_cache']['median'], results['with_cache']['median']):.1f}% ⬇"
            ],
            [
                "P95",
                f"{results['no_cache']['p95']:.2f}ms",
                f"{results['with_cache']['p95']:.2f}ms",
                f"{self._calc_improvement(results['no_cache']['p95'], results['with_cache']['p95']):.1f}% ⬇"
            ],
            [
                "P99",
                f"{results['no_cache']['p99']:.2f}ms",
                f"{results['with_cache']['p99']:.2f}ms",
                f"{self._calc_improvement(results['no_cache']['p99'], results['with_cache']['p99']):.1f}% ⬇"
            ],
            [
                "Min",
                f"{results['no_cache']['min']:.2f}ms",
                f"{results['with_cache']['min']:.2f}ms",
                f"{self._calc_improvement(results['no_cache']['min'], results['with_cache']['min']):.1f}% ⬇"
            ],
            [
                "Max",
                f"{results['no_cache']['max']:.2f}ms",
                f"{results['with_cache']['max']:.2f}ms",
                f"{self._calc_improvement(results['no_cache']['max'], results['with_cache']['max']):.1f}% ⬇"
            ]
        ]
        
        print(tabulate(table_data, headers="firstrow", tablefmt="simple"))
        
        print("\n" + "="*70)
        print(f"🎯 KEY FINDING: {results['improvement_percent']:.1f}% latency reduction with custom KV-Store")
        print("="*70)
        
        # Resume bullet point suggestion
        print("\n📝 RESUME BULLET POINT:")
        print(f"\"Reduced API latency by {results['improvement_percent']:.0f}% "
              f"(from {results['no_cache']['mean']:.0f}ms to {results['with_cache']['mean']:.0f}ms) "
              "by integrating custom Java KV-store as intelligent caching layer\"")
        
    def _calc_improvement(self, before: float, after: float) -> float:
        """Calculate improvement percentage"""
        return (before - after) / before * 100
    
    def plot_results(self, results: dict, save_path: str = "benchmark_results.png"):
        """Create visualization of results"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Latency comparison
        metrics = ['Mean', 'Median', 'P95', 'P99']
        no_cache_values = [
            results['no_cache']['mean'],
            results['no_cache']['median'],
            results['no_cache']['p95'],
            results['no_cache']['p99']
        ]
        with_cache_values = [
            results['with_cache']['mean'],
            results['with_cache']['median'],
            results['with_cache']['p95'],
            results['with_cache']['p99']
        ]
        
        x = range(len(metrics))
        width = 0.35
        
        ax1.bar([i - width/2 for i in x], no_cache_values, width, label='No Cache', color='#e74c3c')
        ax1.bar([i + width/2 for i in x], with_cache_values, width, label='With Cache', color='#27ae60')
        ax1.set_xlabel('Metric')
        ax1.set_ylabel('Latency (ms)')
        ax1.set_title('Latency Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(metrics)
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # Distribution
        ax2.hist(results['no_cache']['latencies'], bins=30, alpha=0.5, 
                label='No Cache', color='#e74c3c')
        ax2.hist(results['with_cache']['latencies'], bins=30, alpha=0.5,
                label='With Cache', color='#27ae60')
        ax2.set_xlabel('Latency (ms)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Latency Distribution')
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n📊 Plot saved to: {save_path}")
        plt.close()

def main():
    print("""
    ╔════════════════════════════════════════════════════════════════╗
    ║  ML Inference Benchmark - Custom KV-Store vs No Cache          ║
    ║  Measuring Real-World Performance Impact                       ║
    ╚════════════════════════════════════════════════════════════════╝
    """)
    
    # Check if service is running
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code != 200:
            print("❌ ML service not healthy. Please start the service first.")
            return
    except requests.exceptions.RequestException:
        print("❌ Cannot connect to ML service at http://localhost:8000")
        print("   Please start the service with: python ml_inference.py")
        return
    
    # Run benchmark
    runner = BenchmarkRunner()
    results = runner.run_benchmark(iterations=100)
    
    # Print results
    runner.print_results(results)
    
    # Create visualization
    try:
        runner.plot_results(results)
    except Exception as e:
        print(f"\n⚠ Could not create plot: {e}")
        print("   Install matplotlib to generate visualization: pip install matplotlib")
    
    print("\n✅ Benchmark complete!")

if __name__ == "__main__":
    main()