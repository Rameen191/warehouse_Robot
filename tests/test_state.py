import pytest
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.state import RobotState
from src.problem import WarehouseProblem


def test_state_equality():
    """Test state equality check."""
    state1 = RobotState((0, 0), frozenset([(1, 1)]))
    state2 = RobotState((0, 0), frozenset([(1, 1)]))
    state3 = RobotState((0, 1), frozenset([(1, 1)]))
    assert state1 == state2
    assert state1 != state3


def test_state_hash():
    """Test state hashing."""
    state1 = RobotState((0, 0), frozenset([(1, 1)]))
    state2 = RobotState((0, 0), frozenset([(1, 1)]))
    assert hash(state1) == hash(state2)


def test_neighbors_basic():
    """Test neighbor generation."""
    grid = [['S', '.'], ['.', 'D']]
    problem = WarehouseProblem(grid, [], (1, 1))
    state = RobotState((0, 0), frozenset())
    neighbors = state.get_neighbors(problem)
    assert len(neighbors) == 2  # Down and Right


def test_package_collection():
    """Test package collection works."""
    grid = [['S', 'P'], ['.', 'D']]
    problem = WarehouseProblem(grid, [(0, 1)], (1, 1))
    state = RobotState((0, 0), frozenset())
    neighbors = state.get_neighbors(problem)

    # Moving to package should collect it
    collected_package_found = False
    for neighbor in neighbors:
        if neighbor.position == (0, 1):
            assert (0, 1) in neighbor.collected
            collected_package_found = True
    assert collected_package_found, "Package collection neighbor not found"