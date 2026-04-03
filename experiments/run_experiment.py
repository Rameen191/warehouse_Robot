import mlflow
import sys
import os
import csv
import time
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils import set_seed, load_config
from src.state import RobotState
from src.search import UniformCostSearch
from src.problem import WarehouseProblem

def parse_grid(grid_string):
    """Parse grid string to 2D list."""
    return [row.split() for row in grid_string.strip('"').split(';')]

def get_grid_from_csv(grid_name, csv_path="data/raw/warehouse_grids.csv"):
    """Load grid data from CSV file."""
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['grid_name'] == grid_name:
                    return parse_grid(row['grid_data'])
        raise ValueError(f"Grid '{grid_name}' not found")
    except FileNotFoundError:
        print(f"Warning: {csv_path} not found")
        return None

def run_experiment(grid_name=None, seed=42):
    """Run a single experiment."""
    print(f"\n{'=' * 60}")
    print(f"Experiment: {grid_name or 'default'} with seed {seed}")
    print('=' * 60)

    # Set seed
    set_seed(seed)

    # Get grid
    grid = get_grid_from_csv(grid_name) if grid_name else get_grid_from_csv("simple_5x5")
    if not grid:
        print("Error: Could not load grid")
        return None, None, None, None

    # Find positions
    package_positions = []
    delivery_pos = None
    start_pos = None

    for i in range(len(grid)):
        for j in range(len(grid[i])):
            if grid[i][j] == 'P':
                package_positions.append((i, j))
            elif grid[i][j] == 'D':
                delivery_pos = (i, j)
            elif grid[i][j] == 'S':
                start_pos = (i, j)

    print(f"Grid: {len(grid)}x{len(grid[0])}")
    print(f"Packages: {len(package_positions)}")

    # Create problem
    problem = WarehouseProblem(grid, package_positions, delivery_pos)
    initial_state = RobotState(start_pos, frozenset())

    # Set up MLflow
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    experiment_name = "warehouse_robot_v1"

    # Create experiment if needed
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        mlflow.create_experiment(experiment_name)

    mlflow.set_experiment(experiment_name)

    # Create run name
    run_name = f"{grid_name or 'simple_5x5'}_seed_{seed}"

    # Run experiment
    with mlflow.start_run(run_name=run_name):
        # Log parameters
        mlflow.log_param("grid_name", grid_name or "simple_5x5")
        mlflow.log_param("grid_size", f"{len(grid)}x{len(grid[0])}")
        mlflow.log_param("num_packages", len(package_positions))
        mlflow.log_param("seed", seed)
        mlflow.log_param("algorithm", "UCS")
        mlflow.set_tag("timestamp", datetime.now().isoformat())

        # Run search
        print("\nRunning UCS search...")
        ucs = UniformCostSearch(problem)
        path, cost, nodes_expanded, time_taken = ucs.solve(initial_state, log_to_mlflow=True)

        # Log metrics
        if path:
            mlflow.log_metric("solution_found", 1)
            mlflow.log_metric("cost", cost)
            mlflow.log_metric("path_length", len(path))
            print(f"\n✓ Solution found!")
            print(f"  Cost: {cost}")
            print(f"  Path length: {len(path)}")
        else:
            mlflow.log_metric("solution_found", 0)
            print(f"\n✗ No solution found")

        mlflow.log_metric("nodes_expanded", nodes_expanded)
        mlflow.log_metric("time_seconds", time_taken)

        # Save solution as artifact
        if path:
            artifact_file = f"solution_{grid_name or 'simple_5x5'}_seed_{seed}.txt"
            with open(artifact_file, "w") as f:
                f.write(f"Grid: {grid_name}\n")
                f.write(f"Seed: {seed}\n")
                f.write(f"Path: {path}\n")
                f.write(f"Cost: {cost}\n")
            mlflow.log_artifact(artifact_file)
            os.remove(artifact_file)

        print(f"\nRun ID: {mlflow.active_run().info.run_id}")
        print(f"View at: http://127.0.0.1:5000/#/experiments/{mlflow.active_run().info.experiment_id}")

        return path, cost, nodes_expanded, time_taken

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", choices=['simple_5x5', 'medium_8x8', 'hard_10x10'])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_experiment(grid_name=args.grid, seed=args.seed)