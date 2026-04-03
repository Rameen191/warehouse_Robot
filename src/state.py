class RobotState:
    def __init__(self, position, collected=frozenset()):
        self.position = position
        self.collected = collected

    def __eq__(self, other):
        return (self.position == other.position and
                self.collected == other.collected)

    def __hash__(self):
        return hash((self.position, self.collected))

    def __repr__(self):
        return f"RobotState(pos={self.position}, collected={len(self.collected)})"

    def is_goal(self, problem):
        """Check if state is goal state."""
        return (self.position == problem.delivery_pos and
                len(self.collected) == len(problem.package_positions))

    def get_neighbors(self, problem):
        """Generate all possible next states."""

        neighbors = []
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right

        for dr, dc in moves:
            new_row = self.position[0] + dr
            new_col = self.position[1] + dc

            # Check bounds
            if not (0 <= new_row < problem.height and 0 <= new_col < problem.width):
                continue

            # Check obstacle
            if problem.grid[new_row][new_col] == 'X':
                continue

            # Check if new position has package
            new_collected = set(self.collected)
            if (new_row, new_col) in problem.package_positions:
                new_collected.add((new_row, new_col))

            neighbors.append(
                RobotState((new_row, new_col), frozenset(new_collected))
            )

        return neighbors