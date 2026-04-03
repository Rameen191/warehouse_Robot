import heapq
import time
import itertools
import mlflow

class RobotState:
    def __init__(self, position, packages_collected):
        self.position = position
        self.packages_collected = packages_collected  # frozenset of collected packages

    def __hash__(self):
        return hash((self.position, self.packages_collected))

    def __eq__(self, other):
        return (self.position, self.packages_collected) == (other.position, other.packages_collected)

    def __lt__(self, other):
        # Needed for heapq tie-breaks; arbitrary comparison
        return False

    def get_neighbors(self, problem):
        neighbors = []
        directions = [(-1,0), (1,0), (0,-1), (0,1)]
        for dx, dy in directions:
            new_pos = (self.position[0]+dx, self.position[1]+dy)
            if problem.is_valid_position(new_pos):
                new_packages = set(self.packages_collected)
                if new_pos in problem.package_positions:
                    new_packages.add(new_pos)
                neighbor = RobotState(new_pos, frozenset(new_packages))
                neighbors.append((neighbor, 1))
        return neighbors


class WarehouseProblem:
    def __init__(self, grid, package_positions, delivery_pos):
        self.grid = grid
        self.package_positions = set(package_positions)
        self.delivery_pos = delivery_pos

    def is_valid_position(self, pos):
        x, y = pos
        return 0 <= x < len(self.grid) and 0 <= y < len(self.grid[0]) and self.grid[x][y] != 'X'

    def is_goal(self, state):
        # Goal: all packages collected and at delivery position
        return state.packages_collected == self.package_positions and state.position == self.delivery_pos


class UniformCostSearch:
    """UCS with tie-breaker and optional MLflow logging."""
    def __init__(self, problem):
        self.problem = problem
        self.nodes_expanded = 0
        self.max_frontier_size = 0

    def solve(self, initial_state, log_to_mlflow=False):
        start_time = time.time()
        frontier = []
        explored = set()
        counter = itertools.count()

        heapq.heappush(frontier, (0, next(counter), initial_state, [initial_state.position]))

        while frontier:
            self.max_frontier_size = max(self.max_frontier_size, len(frontier))
            cost, _, state, path = heapq.heappop(frontier)

            print(f"Expanding: {state.position}, Packages: {state.packages_collected}")  # Debug

            if self.problem.is_goal(state):
                elapsed_time = time.time() - start_time
                print("Goal reached!")

                if log_to_mlflow:
                    self._log_experiment(cost, elapsed_time)

                return path, cost, self.nodes_expanded, elapsed_time

            if state in explored:
                continue

            explored.add(state)
            self.nodes_expanded += 1

            for neighbor, move_cost in state.get_neighbors(self.problem):
                if neighbor not in explored:
                    new_cost = cost + move_cost
                    new_path = path + [neighbor.position]
                    heapq.heappush(frontier, (new_cost, next(counter), neighbor, new_path))

        return None, float('inf'), self.nodes_expanded, 0

    def _log_experiment(self, cost, time_taken):
        mlflow.log_param("algorithm", "UCS")
        mlflow.log_metric("total_cost", cost)
        mlflow.log_metric("nodes_expanded", self.nodes_expanded)
        mlflow.log_metric("max_frontier_size", self.max_frontier_size)
        mlflow.log_metric("time_seconds", time_taken)


# ===== Example usage =====
if __name__ == "__main__":
    # Example grid
    grid = [
        ['S', ' ', ' ', 'P', ' '],
        [' ', 'X', ' ', ' ', ' '],
        [' ', ' ', ' ', 'X', ' '],
        ['P', ' ', ' ', ' ', 'D'],
        [' ', ' ', ' ', ' ', ' '],
    ]

    package_positions = [(0,3), (3,0)]
    delivery_pos = (3,4)
    start_pos = (0,0)

    problem = WarehouseProblem(grid, package_positions, delivery_pos)
    initial_state = RobotState(start_pos, frozenset())

    ucs = UniformCostSearch(problem)
    path, cost, nodes, elapsed = ucs.solve(initial_state)

    if path:
        print("\n✓ Solution found!")
        print(f"Cost: {cost}")
        print(f"Path: {path}")
        print(f"Nodes expanded: {nodes}")
        print(f"Time: {elapsed:.3f}s")
    else:
        print("\n✗ No solution found")
