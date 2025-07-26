"""
Helper functions for displaying results and analyzing optimization performance
"""

import json
from datetime import datetime
from typing import Optional

from models import OptimizationResult


def display_optimization_results(result: OptimizationResult):
    """Display optimization results in a formatted way"""
    print("\n" + "=" * 60)
    print("🎯 PROMPT OPTIMIZATION RESULTS")
    print("=" * 60)

    if result.success:
        print(f"✅ Status: SUCCESS")
        print(f"📊 Final Score: {result.final_score:.3f}")
        print(f"🔄 Iterations: {result.iterations_completed}")
        print(f"📈 Total Versions: {len(result.all_versions)}")

        print("\n" + "-" * 40)
        print("🏆 BEST PROMPT:")
        print("-" * 40)
        print(result.final_prompt)

        print("\n" + "-" * 40)
        print("💭 EVALUATION REASONING:")
        print("-" * 40)
        print(result.reasoning)

        if len(result.all_versions) > 1:
            print("\n" + "-" * 40)
            print("📈 VERSION HISTORY:")
            print("-" * 40)
            for version in result.all_versions:
                print(f"Version {version.version}: {version.average_score:.3f}")
    else:
        print(f"❌ Status: FAILED")
        print(f"💔 Error: {result.error_message}")

    print("\n" + "=" * 60)


def analyze_optimization_results(result: OptimizationResult):
    """Analyze and provide insights on optimization results"""

    if not result.success:
        print("❌ Cannot analyze failed optimization")
        return

    print("\n📊 OPTIMIZATION ANALYSIS")
    print("=" * 40)

    # Performance analysis
    scores = [
        v.average_score for v in result.all_versions if v.average_score is not None
    ]
    if len(scores) > 1:
        improvement = scores[-1] - scores[0]
        print(f"📈 Score Improvement: {improvement:+.3f}")
        print(f"🎯 Final Score: {scores[-1]:.3f}")
        print(f"🥇 Best Score: {max(scores):.3f}")

        if improvement > 0.1:
            print("✅ Significant improvement achieved!")
        elif improvement > 0:
            print("✅ Moderate improvement achieved")
        else:
            print("⚠️ No improvement or score decreased")

    # Iteration analysis
    print(f"\n🔄 Iterations Used: {result.iterations_completed}")
    if result.iterations_completed == 1:
        print("🎯 Target achieved on first try!")
    elif result.final_score >= 0.8:
        print("🏆 Excellent final score achieved")
    elif result.final_score >= 0.6:
        print("✅ Good final score achieved")
    else:
        print("⚠️ Score below typical targets - consider more iterations")


def export_results_to_json(
    result: OptimizationResult, filename: Optional[str] = None
) -> Optional[str]:
    """Export optimization results to JSON file"""

    if filename is None:
        filename = f"results/prompt_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    try:
        # Convert to dict for JSON serialization
        result_dict = result.dict()

        # Add metadata
        result_dict["export_metadata"] = {
            "exported_at": datetime.now().isoformat(),
            "notebook_version": "1.0",
            "workflow_type": "modular_prompt_optimizer",
        }

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(result_dict, f, indent=2, default=str)

        print(f"✅ Results exported to: {filename}")
        return filename

    except Exception as e:
        print(f"❌ Export failed: {e}")
        return None
