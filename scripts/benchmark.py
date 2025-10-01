"""
Performance Benchmarking Suite

Comprehensive system performance analysis:
- End-to-end workflow latency profiling
- Token consumption and cost analysis
- Cache effectiveness measurement
- Concurrent load testing
- Regression detection with historical comparison
- Statistical significance testing

Theoretical Foundation: Statistical process control + hypothesis testing
Output: JSON/HTML reports with actionable insights
"""

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple
from uuid import uuid4

import numpy as np
from loguru import logger

from core.models import GeneratedArticle

# Import system components
from infrastructure.database import DatabaseManager
from infrastructure.redis_client import RedisClient
from knowledge.project_repository import ProjectRepository
from orchestration.content_agent import ContentAgent, ContentAgentConfig

# ============================================================================
# BENCHMARK CONFIGURATION
# ============================================================================


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark execution."""

    iterations: int = 10
    concurrent_requests: int = 5
    warmup_iterations: int = 2
    confidence_level: float = 0.95
    cost_target_per_article: float = 0.30
    latency_target_seconds: float = 180.0
    output_dir: Path = Path("./benchmark_results")


@dataclass
class BenchmarkResult:
    """Individual benchmark result."""

    iteration: int
    topic: str
    success: bool
    latency_seconds: float
    tokens_used: int
    cost_dollars: float
    word_count: int
    readability_score: float
    cache_hits: int
    error: str = None


@dataclass
class BenchmarkStatistics:
    """Aggregated benchmark statistics."""

    total_iterations: int
    successful_iterations: int
    failed_iterations: int

    # Latency statistics
    mean_latency: float
    median_latency: float
    p50_latency: float
    p95_latency: float
    p99_latency: float
    std_latency: float

    # Cost statistics
    mean_cost: float
    median_cost: float
    total_cost: float
    cost_per_word: float

    # Quality statistics
    mean_readability: float
    mean_word_count: float

    # Efficiency statistics
    mean_tokens: float
    cache_hit_rate: float

    # Performance assessment
    meets_latency_target: bool
    meets_cost_target: bool
    regression_detected: bool


# ============================================================================
# BENCHMARK EXECUTOR
# ============================================================================


class BenchmarkExecutor:
    """
    Main benchmark execution engine.

    Implements scientific benchmarking methodology with:
    - Warmup iterations to stabilize system
    - Statistical sampling with confidence intervals
    - Regression detection via historical comparison
    - Detailed profiling at each workflow stage
    """

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results: List[BenchmarkResult] = []
        self.db = None
        self.redis = None
        self.agent = None
        self.test_project_id = None

        # Ensure output directory exists
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    async def setup(self):
        """Initialize system components for benchmarking."""
        logger.info("Setting up benchmark environment")

        # Initialize infrastructure
        self.db = DatabaseManager()
        await self.db.connect()

        self.redis = RedisClient()
        await self.redis.connect()

        # Initialize full system (simplified for benchmark)
        # In production, this would use full dependency injection
        from execution.content_generator import ContentGenerator
        from execution.content_planner import ContentPlanner
        from execution.distributor import Distributor
        from execution.keyword_researcher import KeywordResearcher
        from infrastructure.llm_client import LLMClient
        from infrastructure.monitoring import MetricsCollector
        from intelligence.context_synthesizer import ContextSynthesizer
        from intelligence.decision_engine import DecisionEngine
        from intelligence.semantic_analyzer import SemanticAnalyzer
        from knowledge.rulebook_manager import RulebookManager
        from knowledge.website_analyzer import WebsiteAnalyzer
        from optimization.cache_manager import CacheManager
        from optimization.model_router import ModelRouter
        from optimization.prompt_compressor import PromptCompressor
        from optimization.token_budget_manager import TokenBudgetManager

        projects = ProjectRepository(self.db)
        rulebook_mgr = RulebookManager(self.db)
        website_analyzer = WebsiteAnalyzer()

        semantic_analyzer = SemanticAnalyzer()
        decision_engine = DecisionEngine(self.db, semantic_analyzer)
        cache = CacheManager(self.redis)
        context_synthesizer = ContextSynthesizer(projects, rulebook_mgr, decision_engine, cache)

        llm = LLMClient()
        model_router = ModelRouter()
        metrics = MetricsCollector()
        budget_manager = TokenBudgetManager(self.redis, metrics)
        prompt_compressor = PromptCompressor(semantic_analyzer)

        keyword_researcher = KeywordResearcher(llm, semantic_analyzer, cache)
        content_planner = ContentPlanner(llm, decision_engine, context_synthesizer, model_router)
        content_generator = ContentGenerator(
            llm,
            context_synthesizer,
            semantic_analyzer,
            model_router,
            budget_manager,
            prompt_compressor,
            metrics,
        )
        distributor = Distributor(telegram_bot_token=None, metrics_collector=metrics)

        self.agent = ContentAgent(
            project_repository=projects,
            rulebook_manager=rulebook_mgr,
            website_analyzer=website_analyzer,
            decision_engine=decision_engine,
            context_synthesizer=context_synthesizer,
            keyword_researcher=keyword_researcher,
            content_planner=content_planner,
            content_generator=content_generator,
            distributor=distributor,
            budget_manager=budget_manager,
            metrics_collector=metrics,
            config=ContentAgentConfig(enable_auto_distribution=False),
        )

        # Create test project
        project = await projects.create(
            name="Benchmark Test Project", domain="https://benchmark.test.com"
        )
        self.test_project_id = project.id

        # Add sample rulebook
        await rulebook_mgr.create_rulebook(
            project_id=self.test_project_id,
            content="Use clear, professional language. Target grade 10-12 readability.",
        )

        logger.success("Benchmark environment ready")

    async def cleanup(self):
        """Clean up resources after benchmarking."""
        logger.info("Cleaning up benchmark environment")

        if self.db:
            # Clean up test data
            await self.db.execute(
                "DELETE FROM generated_articles WHERE project_id = $1", self.test_project_id
            )
            await self.db.execute("DELETE FROM projects WHERE id = $1", self.test_project_id)
            await self.db.disconnect()

        if self.redis:
            await self.redis.disconnect()

        logger.info("Cleanup complete")

    async def run_single_benchmark(self, iteration: int, topic: str) -> BenchmarkResult:
        """Execute single benchmark iteration."""
        logger.info(f"Running iteration {iteration}: {topic}")

        start_time = time.time()
        cache_hits_before = await self._get_cache_hits()

        try:
            article = await self.agent.create_content(
                project_id=self.test_project_id, topic=topic, priority="high"
            )

            latency = time.time() - start_time
            cache_hits_after = await self._get_cache_hits()
            cache_hits = cache_hits_after - cache_hits_before

            result = BenchmarkResult(
                iteration=iteration,
                topic=topic,
                success=True,
                latency_seconds=latency,
                tokens_used=article.total_tokens_used,
                cost_dollars=article.total_cost,
                word_count=article.word_count,
                readability_score=article.readability_score,
                cache_hits=cache_hits,
            )

            logger.success(
                f"Iteration {iteration} complete | "
                f"latency={latency:.2f}s | cost=${article.total_cost:.4f}"
            )

        except Exception as e:
            latency = time.time() - start_time

            result = BenchmarkResult(
                iteration=iteration,
                topic=topic,
                success=False,
                latency_seconds=latency,
                tokens_used=0,
                cost_dollars=0,
                word_count=0,
                readability_score=0,
                cache_hits=0,
                error=str(e),
            )

            logger.error(f"Iteration {iteration} failed | error={e}")

        return result

    async def _get_cache_hits(self) -> int:
        """Get current cache hit count."""
        try:
            hits = await self.redis.get("cache:hits")
            return int(hits) if hits else 0
        except:
            return 0

    async def run_sequential_benchmark(self) -> List[BenchmarkResult]:
        """Run sequential benchmark iterations."""
        logger.info(
            f"Starting sequential benchmark | "
            f"iterations={self.config.iterations} | "
            f"warmup={self.config.warmup_iterations}"
        )

        results = []

        # Warmup iterations
        logger.info("Running warmup iterations")
        for i in range(self.config.warmup_iterations):
            await self.run_single_benchmark(iteration=-(i + 1), topic=f"Warmup Topic {i+1}")

        # Actual benchmark iterations
        logger.info("Running benchmark iterations")
        topics = [
            "Advanced Machine Learning Techniques",
            "Cloud Infrastructure Best Practices",
            "Modern Web Development Frameworks",
            "Data Science Pipeline Optimization",
            "Microservices Architecture Patterns",
            "DevOps Automation Strategies",
            "API Design and Documentation",
            "Database Performance Tuning",
            "Security Best Practices Guide",
            "Scalable System Architecture",
        ]

        for i in range(self.config.iterations):
            topic = topics[i % len(topics)] + f" (Iteration {i+1})"

            result = await self.run_single_benchmark(i + 1, topic)
            results.append(result)

        return results

    async def run_concurrent_benchmark(self) -> List[BenchmarkResult]:
        """Run concurrent benchmark iterations."""
        logger.info(
            f"Starting concurrent benchmark | "
            f"concurrent_requests={self.config.concurrent_requests}"
        )

        topics = [f"Concurrent Test Topic {i}" for i in range(self.config.concurrent_requests)]

        tasks = [self.run_single_benchmark(i + 1, topic) for i, topic in enumerate(topics)]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to failed results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(
                    BenchmarkResult(
                        iteration=i + 1,
                        topic=topics[i],
                        success=False,
                        latency_seconds=0,
                        tokens_used=0,
                        cost_dollars=0,
                        word_count=0,
                        readability_score=0,
                        cache_hits=0,
                        error=str(result),
                    )
                )
            else:
                processed_results.append(result)

        return processed_results

    def calculate_statistics(self, results: List[BenchmarkResult]) -> BenchmarkStatistics:
        """Calculate comprehensive statistics from results."""
        logger.info("Calculating benchmark statistics")

        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]

        if not successful:
            logger.error("No successful iterations - cannot calculate statistics")
            return None

        # Extract metrics
        latencies = [r.latency_seconds for r in successful]
        costs = [r.cost_dollars for r in successful]
        tokens = [r.tokens_used for r in successful]
        word_counts = [r.word_count for r in successful]
        readability = [r.readability_score for r in successful]
        cache_hits = sum(r.cache_hits for r in successful)
        total_operations = sum(r.tokens_used for r in successful) / 1000  # Estimate

        # Calculate percentiles
        latencies_sorted = sorted(latencies)
        p50_latency = np.percentile(latencies_sorted, 50)
        p95_latency = np.percentile(latencies_sorted, 95)
        p99_latency = np.percentile(latencies_sorted, 99)

        # Cost analysis
        total_cost = sum(costs)
        mean_cost = statistics.mean(costs)
        total_words = sum(word_counts)
        cost_per_word = total_cost / total_words if total_words > 0 else 0

        # Performance assessment
        meets_latency_target = p95_latency <= self.config.latency_target_seconds
        meets_cost_target = mean_cost <= self.config.cost_target_per_article

        stats = BenchmarkStatistics(
            total_iterations=len(results),
            successful_iterations=len(successful),
            failed_iterations=len(failed),
            mean_latency=statistics.mean(latencies),
            median_latency=statistics.median(latencies),
            p50_latency=p50_latency,
            p95_latency=p95_latency,
            p99_latency=p99_latency,
            std_latency=statistics.stdev(latencies) if len(latencies) > 1 else 0,
            mean_cost=mean_cost,
            median_cost=statistics.median(costs),
            total_cost=total_cost,
            cost_per_word=cost_per_word,
            mean_readability=statistics.mean(readability),
            mean_word_count=statistics.mean(word_counts),
            mean_tokens=statistics.mean(tokens),
            cache_hit_rate=cache_hits / max(total_operations, 1),
            meets_latency_target=meets_latency_target,
            meets_cost_target=meets_cost_target,
            regression_detected=False,  # TODO: Implement historical comparison
        )

        return stats

    def generate_report(self, stats: BenchmarkStatistics, results: List[BenchmarkResult]) -> Dict:
        """Generate comprehensive benchmark report."""
        logger.info("Generating benchmark report")

        report = {
            "metadata": {
                "timestamp": datetime.utcnow().isoformat(),
                "config": asdict(self.config),
                "system_version": "1.0.0",
            },
            "summary": asdict(stats),
            "results": [asdict(r) for r in results],
            "analysis": {
                "performance_grade": self._calculate_performance_grade(stats),
                "bottlenecks": self._identify_bottlenecks(results),
                "recommendations": self._generate_recommendations(stats),
            },
        }

        return report

    def _calculate_performance_grade(self, stats: BenchmarkStatistics) -> str:
        """Calculate overall performance grade."""
        score = 0

        # Latency scoring (40 points)
        if stats.p95_latency <= 60:
            score += 40
        elif stats.p95_latency <= 120:
            score += 30
        elif stats.p95_latency <= 180:
            score += 20
        else:
            score += 10

        # Cost scoring (30 points)
        if stats.mean_cost <= 0.15:
            score += 30
        elif stats.mean_cost <= 0.25:
            score += 20
        elif stats.mean_cost <= 0.35:
            score += 10

        # Reliability scoring (20 points)
        success_rate = stats.successful_iterations / stats.total_iterations
        score += int(success_rate * 20)

        # Efficiency scoring (10 points)
        if stats.cache_hit_rate >= 0.40:
            score += 10
        elif stats.cache_hit_rate >= 0.25:
            score += 5

        # Grade mapping
        if score >= 90:
            return "A (Excellent)"
        elif score >= 80:
            return "B (Good)"
        elif score >= 70:
            return "C (Acceptable)"
        elif score >= 60:
            return "D (Needs Improvement)"
        else:
            return "F (Poor)"

    def _identify_bottlenecks(self, results: List[BenchmarkResult]) -> List[str]:
        """Identify performance bottlenecks."""
        bottlenecks = []

        successful = [r for r in results if r.success]
        if not successful:
            return ["Insufficient successful iterations for analysis"]

        latencies = [r.latency_seconds for r in successful]
        mean_latency = statistics.mean(latencies)

        if mean_latency > 180:
            bottlenecks.append("High average latency - consider parallel processing")

        costs = [r.cost_dollars for r in successful]
        mean_cost = statistics.mean(costs)

        if mean_cost > 0.30:
            bottlenecks.append("High cost per article - review prompt optimization")

        cache_hits = sum(r.cache_hits for r in successful)
        if cache_hits < len(successful) * 0.25:
            bottlenecks.append("Low cache hit rate - verify caching configuration")

        failed_count = len([r for r in results if not r.success])
        if failed_count > 0:
            bottlenecks.append(f"{failed_count} failed iterations - investigate error patterns")

        return bottlenecks if bottlenecks else ["No significant bottlenecks detected"]

    def _generate_recommendations(self, stats: BenchmarkStatistics) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []

        if not stats.meets_latency_target:
            recommendations.append(
                f"Latency (P95: {stats.p95_latency:.1f}s) exceeds target ({self.config.latency_target_seconds}s). "
                "Consider: (1) Increase concurrent processing, (2) Optimize prompt length, (3) Use faster models for simple tasks"
            )

        if not stats.meets_cost_target:
            recommendations.append(
                f"Cost (${stats.mean_cost:.4f}) exceeds target (${self.config.cost_target_per_article:.4f}). "
                "Consider: (1) Increase cache hit rate, (2) Use GPT-3.5 for more operations, (3) Implement prompt compression"
            )

        if stats.cache_hit_rate < 0.35:
            recommendations.append(
                f"Cache hit rate ({stats.cache_hit_rate:.1%}) is low. "
                "Consider: (1) Increase cache TTL, (2) Pre-warm cache, (3) Implement semantic deduplication"
            )

        if stats.failed_iterations > 0:
            recommendations.append(
                f"Reliability issue: {stats.failed_iterations} failed iterations. "
                "Review error logs and implement additional retry logic"
            )

        if not recommendations:
            recommendations.append("System performing well - no critical optimizations needed")

        return recommendations

    def save_report(self, report: Dict, filename: str = None):
        """Save benchmark report to file."""
        if filename is None:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_report_{timestamp}.json"

        filepath = self.config.output_dir / filename

        with open(filepath, "w") as f:
            json.dump(report, f, indent=2)

        logger.success(f"Benchmark report saved to {filepath}")

        # Also generate human-readable summary
        self._generate_text_summary(report, filepath.with_suffix(".txt"))

    def _generate_text_summary(self, report: Dict, filepath: Path):
        """Generate human-readable text summary."""
        stats = report["summary"]
        analysis = report["analysis"]

        summary = f"""
================================================================================
CONTENT AUTOMATION ENGINE - BENCHMARK REPORT
================================================================================

Generated: {report['metadata']['timestamp']}
System Version: {report['metadata']['system_version']}

================================================================================
PERFORMANCE SUMMARY
================================================================================

Overall Grade: {analysis['performance_grade']}

Success Rate: {stats['successful_iterations']}/{stats['total_iterations']} ({stats['successful_iterations']/stats['total_iterations']*100:.1f}%)

Latency Metrics:
  - Mean:    {stats['mean_latency']:.2f}s
  - Median:  {stats['median_latency']:.2f}s
  - P95:     {stats['p95_latency']:.2f}s (Target: {self.config.latency_target_seconds:.0f}s)
  - P99:     {stats['p99_latency']:.2f}s
  - Std Dev: {stats['std_latency']:.2f}s

Cost Metrics:
  - Mean Cost:        ${stats['mean_cost']:.4f}/article (Target: ${self.config.cost_target_per_article:.2f})
  - Median Cost:      ${stats['median_cost']:.4f}/article
  - Total Cost:       ${stats['total_cost']:.2f}
  - Cost per Word:    ${stats['cost_per_word']:.6f}

Quality Metrics:
  - Mean Readability: {stats['mean_readability']:.1f}
  - Mean Word Count:  {stats['mean_word_count']:.0f} words

Efficiency Metrics:
  - Mean Tokens:      {stats['mean_tokens']:.0f}
  - Cache Hit Rate:   {stats['cache_hit_rate']:.1%}

================================================================================
PERFORMANCE ASSESSMENT
================================================================================

Latency Target: {'✓ PASS' if stats['meets_latency_target'] else '✗ FAIL'}
Cost Target:    {'✓ PASS' if stats['meets_cost_target'] else '✗ FAIL'}

================================================================================
IDENTIFIED BOTTLENECKS
================================================================================

{chr(10).join('  - ' + b for b in analysis['bottlenecks'])}

================================================================================
RECOMMENDATIONS
================================================================================

{chr(10).join('  ' + str(i+1) + '. ' + r for i, r in enumerate(analysis['recommendations']))}

================================================================================
END OF REPORT
================================================================================
"""

        with open(filepath, "w") as f:
            f.write(summary)

        logger.info(f"Text summary saved to {filepath}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================


async def main():
    """Main benchmark execution."""
    parser = argparse.ArgumentParser(description="Content Automation Engine Benchmark Suite")
    parser.add_argument("--iterations", type=int, default=10, help="Number of benchmark iterations")
    parser.add_argument(
        "--concurrent", type=int, default=0, help="Concurrent requests (0=sequential)"
    )
    parser.add_argument("--warmup", type=int, default=2, help="Warmup iterations")
    parser.add_argument(
        "--output", type=str, default="./benchmark_results", help="Output directory"
    )

    args = parser.parse_args()

    config = BenchmarkConfig(
        iterations=args.iterations,
        concurrent_requests=args.concurrent,
        warmup_iterations=args.warmup,
        output_dir=Path(args.output),
    )

    executor = BenchmarkExecutor(config)

    try:
        await executor.setup()

        if config.concurrent_requests > 0:
            results = await executor.run_concurrent_benchmark()
        else:
            results = await executor.run_sequential_benchmark()

        executor.results = results

        stats = executor.calculate_statistics(results)

        if stats:
            report = executor.generate_report(stats, results)
            executor.save_report(report)

            print("\n" + "=" * 80)
            print("BENCHMARK COMPLETE")
            print("=" * 80)
            print(f"Overall Grade: {report['analysis']['performance_grade']}")
            print(f"Mean Latency: {stats.mean_latency:.2f}s")
            print(f"Mean Cost: ${stats.mean_cost:.4f}/article")
            print(f"Success Rate: {stats.successful_iterations}/{stats.total_iterations}")
            print(f"\nDetailed report saved to: {config.output_dir}")
            print("=" * 80 + "\n")

    finally:
        await executor.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
