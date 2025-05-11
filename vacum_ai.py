"""
Vacuum Cleaner AI Simulation
This program simulates a vacuum cleaner agent that uses A* pathfinding algorithm
and TSP optimization to efficiently clean dirty cells in a grid environment.

Key Features:
- Grid-based environment with randomly placed dirty cells
- A* pathfinding with improved heuristics
- TSP optimization for finding shortest cleaning path
- Smart start position selection for optimal cleaning
- Movement constraints: forward, left, right (no backward or diagonal movement)

Algorithms Used:
1. A* Pathfinding: For finding the shortest path between cells
2. TSP (Traveling Salesman Problem): For optimizing the cleaning sequence
3. Greedy Best-First Search: For initial path planning
4. Priority Queue: For efficient A* implementation
"""

import random
import numpy as np
from queue import PriorityQueue
from itertools import permutations

# Constants for the environment
GRID_SIZE = 3  # Size of the grid (3x3)
DIRTY_CELLS_COUNT = random.randint(3, 5)  # Random number of dirty cells

# Movement directions: forward = up (0,1), left = (-1,0), right = (1,0)
# Vacuum cannot move backward or diagonally
DIRECTIONS = [(0, 1), (-1, 0), (1, 0)]


class VacuumEnvironment:
    """
    Represents the environment where the vacuum cleaner operates.
    Manages the grid state and dirty cell operations.
    """
    def __init__(self):
        # Initialize empty grid (0 = clean, 1 = dirty)
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
        self.place_dirty_cells()

    def place_dirty_cells(self):
        """Randomly place dirty cells in the grid"""
        cells = [(x, y) for x in range(GRID_SIZE) for y in range(GRID_SIZE)]
        dirty_cells = random.sample(cells, DIRTY_CELLS_COUNT)
        for (x, y) in dirty_cells:
            self.grid[y][x] = 1

    def is_dirty(self, x, y):
        """Check if a cell is dirty"""
        return self.grid[y][x] == 1

    def clean_cell(self, x, y):
        """Clean a cell (set to 0)"""
        self.grid[y][x] = 0

    def all_clean(self):
        """Check if all cells are clean"""
        return np.all(self.grid == 0)

    def get_dirty_cells(self):
        """Get list of all dirty cells"""
        return [(x, y) for y in range(GRID_SIZE) for x in range(GRID_SIZE) if self.is_dirty(x, y)]

    def print_grid(self):
        """Print the current state of the grid"""
        print("\nInitial Environment State (D = Dirty Cell):")
        # Print column numbers
        print("    " + "   ".join(str(i) for i in range(GRID_SIZE)) + "  (X)")
        print("  +" + "---+" * GRID_SIZE)
        
        # Print rows with row numbers
        for y in range(GRID_SIZE):
            row = [str(GRID_SIZE - y - 1) + " |"]
            for x in range(GRID_SIZE):
                if self.grid[y][x] == 1:
                    row.append(" D |")
                else:
                    row.append(" . |")
            print(" ".join(row))
            print("  +" + "---+" * GRID_SIZE)
        print("(Y)")


class VacuumAgent:
    """
    Represents the vacuum cleaner agent.
    Uses A* algorithm for pathfinding and TSP for optimal cleaning sequence.
    """
    def __init__(self, env):
        self.env = env
        self.path = []  # Stores the complete path taken
        self.actions = []  # Stores the actions taken
        self.current_pos = None
        self.distance_cache = {}  # Cache for path distances

    def heuristic(self, a, b):
        """
        Improved heuristic for A* algorithm
        Uses Manhattan distance with diagonal movement consideration
        """
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        return max(dx, dy) + (0.414 * min(dx, dy))  # 0.414 is approximately sqrt(2) - 1

    def neighbors(self, node):
        """Get valid neighboring cells based on movement constraints"""
        neighbors = []
        for dx, dy in DIRECTIONS:
            nx, ny = node[0] + dx, node[1] + dy
            if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                neighbors.append((nx, ny))
        return neighbors

    def astar(self, start, goal):
        """
        A* pathfinding algorithm implementation with improved efficiency
        Returns the shortest path from start to goal
        """
        # Check cache first
        cache_key = (start, goal)
        if cache_key in self.distance_cache:
            return self.distance_cache[cache_key]

        frontier = PriorityQueue()
        frontier.put((0, start))
        came_from = {}
        cost_so_far = {}
        came_from[start] = None
        cost_so_far[start] = 0

        while not frontier.empty():
            current = frontier.get()[1]

            if current == goal:
                break

            for next_node in self.neighbors(current):
                new_cost = cost_so_far[current] + 1
                if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                    cost_so_far[next_node] = new_cost
                    priority = new_cost + self.heuristic(goal, next_node)
                    frontier.put((priority, next_node))
                    came_from[next_node] = current

        # Reconstruct path
        current = goal
        path = []
        while current != start:
            path.append(current)
            current = came_from.get(current)
            if current is None:
                return []

        path.reverse()
        # Cache the result
        self.distance_cache[cache_key] = path
        return path

    def calculate_path_length(self, path):
        """Calculate total length of a path"""
        return len(path)

    def find_optimal_sequence(self, start, dirty_cells):
        """
        Find optimal cleaning sequence using TSP approach
        Uses nearest neighbor algorithm for initial solution
        Then tries to improve it with 2-opt local search
        """
        if not dirty_cells:
            return []

        # Start with nearest neighbor solution
        current = start
        unvisited = set(dirty_cells)
        sequence = []

        while unvisited:
            nearest = min(unvisited, key=lambda x: len(self.astar(current, x)))
            sequence.append(nearest)
            unvisited.remove(nearest)
            current = nearest

        # Try to improve the solution with 2-opt
        improved = True
        while improved:
            improved = False
            for i in range(len(sequence) - 1):
                for j in range(i + 1, len(sequence)):
                    # Try swapping two edges
                    new_sequence = sequence[:i] + sequence[i:j+1][::-1] + sequence[j+1:]
                    if self.calculate_total_distance(start, new_sequence) < self.calculate_total_distance(start, sequence):
                        sequence = new_sequence
                        improved = True
                        break
                if improved:
                    break

        return sequence

    def calculate_total_distance(self, start, sequence):
        """Calculate total distance for a cleaning sequence"""
        total = 0
        current = start
        for target in sequence:
            path = self.astar(current, target)
            if path:
                total += len(path)
                current = target
            else:
                return float('inf')
        return total

    def select_best_start(self):
        """
        Find the optimal starting position that minimizes total cleaning time
        Uses TSP optimization for each possible start position
        """
        dirty_cells = self.env.get_dirty_cells()
        if not dirty_cells:
            return (0, 0), []

        candidates = [(x, y) for x in range(GRID_SIZE) for y in range(GRID_SIZE)]
        best_start = None
        best_cost = float('inf')
        best_plan = []

        for start in candidates:
            plan = self.find_optimal_sequence(start, dirty_cells)
            if not plan:
                continue
            cost = self.calculate_total_distance(start, plan)
            if cost < best_cost:
                best_cost = cost
                best_start = start
                best_plan = plan

        return best_start, best_plan

    def execute_plan(self, start, plan):
        """
        Execute the cleaning plan and record all actions
        Returns the complete path and actions taken
        """
        self.current_pos = start
        self.path = [start]
        self.actions = []

        print(f"\nStarting from cell: ({start[0]}, {start[1]})")
        print(f"X-coordinate: {start[0]}, Y-coordinate: {start[1]}")
        
        if self.env.is_dirty(*start):
            print(f"Starting cell is dirty, cleaning first (suck action).")
            self.env.clean_cell(*start)
            self.actions.append('suck')
        else:
            self.actions.append('wait')

        for target in plan:
            if target == start:
                continue
            moves = self.astar(self.current_pos, target)
            print(f"\nMoving from ({self.current_pos[0]}, {self.current_pos[1]}) to ({target[0]}, {target[1]}):")
            for pos in moves:
                print(f"  Moving to cell ({pos[0]}, {pos[1]})")
                self.path.append(pos)
                self.actions.append('move')
                self.current_pos = pos
            if self.env.is_dirty(*self.current_pos):
                print(f"  Cleaning cell ({self.current_pos[0]}, {self.current_pos[1]}) (suck action)")
                self.env.clean_cell(*self.current_pos)
                self.actions.append('suck')
            else:
                self.actions.append('wait')

        return self.path, self.actions


def main():
    """
    Main function to run the vacuum cleaner simulation
    """
    env = VacuumEnvironment()
    env.print_grid()

    agent = VacuumAgent(env)
    start_pos, plan = agent.select_best_start()

    print("\nSimulation Details:")
    print(f"Number of dirty cells: {DIRTY_CELLS_COUNT}")
    print(f"Optimal starting position: ({start_pos[0]}, {start_pos[1]})")
    print("\nCleaning sequence:")
    for i, pos in enumerate(plan, 1):
        print(f"{i}. Cell ({pos[0]}, {pos[1]})")
    
    if plan:
        print(f"\nEnd position: ({plan[-1][0]}, {plan[-1][1]})")
    else:
        print("\nNo dirty cells to clean.")

    print("\nExecuting cleaning plan...")
    agent.execute_plan(start_pos, plan)
    print("\nCleaning Complete!")


if __name__ == "__main__":
    main()