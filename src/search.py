import heapq
import time
import itertools  # for tie-breaker

class UniformCostSearch:
    """Uniform Cost Search implementation."""

    def __init__(self, problem):
        self.problem = problem
        self.nodes_expanded = 0
        self.max_frontier_size = 0

    def solve(self, initial_state):
        start_time = time.time()
        frontier = []
        explored = set()
        counter = itertools.count()  # tie-breaker for heap

        # Push initial state with counter
        heapq.heappush(frontier, (0, next(counter), initial_state, []))

        while frontier:
            cost, _, state, path = heapq.heappop(frontier)

            if self.problem.is_goal(state):
                time_taken = time.time() - start_time
                return path + [state.position], cost, self.nodes_expanded, time_taken

            if state in explored:
                continue

            explored.add(state)
            self.nodes_expanded += 1

            for neighbor, move_cost in state.get_neighbors(self.problem):
                if neighbor not in explored:
                    new_cost = cost + move_cost
                    new_path = path + [state.position]
                    # Push with counter to avoid comparison of RobotState
                    heapq.heappush(frontier, (new_cost, next(counter), neighbor, new_path))

            self.max_frontier_size = max(self.max_frontier_size, len(frontier))

        # No solution found
        time_taken = time.time() - start_time
        return None, None, self.nodes_expanded, time_taken