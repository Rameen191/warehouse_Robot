import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from experiments.run_experiment import run_experiment


def run_variants():
    """Run multiple experiment variants."""

    print("=" * 60)
    print("Running Multiple Experiment Variants")
    print("=" * 60)

    variants = [
        {"grid": "simple_5x5", "seed": 42},
        {"grid": "simple_5x5", "seed": 123},
        {"grid": "medium_8x8", "seed": 42},
        {"grid": "hard_10x10", "seed": 42},
    ]

    results = []

    for i, variant in enumerate(variants, 1):
        print(f"\n[{i}/{len(variants)}] Running {variant['grid']} with seed {variant['seed']}")
        path, cost, nodes, time_taken = run_experiment(
            grid_name=variant['grid'],
            seed=variant['seed']
        )

        results.append({
            'grid': variant['grid'],
            'seed': variant['seed'],
            'found': path is not None,
            'cost': cost,
            'nodes': nodes,
            'time': time_taken
        })

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Grid':<15} {'Seed':<8} {'Status':<10} {'Cost':<8} {'Nodes':<10}")
    print("-" * 60)

    for r in results:
        status = "✓ Found" if r['found'] else "✗ Not found"
        cost = str(r['cost']) if r['found'] else "N/A"
        nodes = str(r['nodes']) if r['nodes'] else "N/A"
        print(f"{r['grid']:<15} {r['seed']:<8} {status:<10} {cost:<8} {nodes:<10}")

    print("=" * 60)


if __name__ == "__main__":
    run_variants()