class WarehouseProblem:
    """Problem formulation for robot delivery task."""

    def __init__(self, grid, package_positions, delivery_pos):
        self.grid = grid
        self.package_positions = set(package_positions)
        self.delivery_pos = delivery_pos
        self.height = len(grid)
        self.width = len(grid[0])

    def get_initial_state(self):
        """Return initial state (position, packages collected)."""

        for i in range(self.height):
            for j in range(self.width):
                if self.grid[i][j] == 'S':
                    return (i, j), frozenset()

        raise ValueError("Start position not found")