# Vacuum Cleaner AI Simulation

An intelligent vacuum cleaner agent that uses A* pathfinding and TSP optimization to find the shortest path for cleaning a grid environment. The agent efficiently navigates through dirty cells while considering movement constraints and optimizing the cleaning sequence.

## Features

- Grid-based environment simulation
- Intelligent path planning using A* algorithm
- TSP optimization for finding the shortest cleaning path
- Smart start position selection
- Real-time cleaning sequence visualization
- Movement constraints (forward, left, right only)

## Algorithms Used

1. **A* Pathfinding**
   - Manhattan distance heuristic
   - Path caching for efficiency
   - Optimized for grid movement

2. **TSP (Traveling Salesman Problem)**
   - Nearest Neighbor algorithm for initial solution
   - 2-opt local search for path optimization
   - Efficient sequence generation

3. **Greedy Best-First Search**
   - Used for initial path planning
   - Optimized for grid-based movement

## Requirements

- Python 3.6+
- NumPy
- Matplotlib (for visualization)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/vacuum-ai.git
cd vacuum-ai
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

Run the simulation:
```bash
python vacum_ai.py
```

The program will:
1. Generate a random grid with dirty cells
2. Find the optimal starting position
3. Calculate the best cleaning sequence
4. Execute the cleaning plan
5. Display the results in the terminal

## Output Format

The program displays:
- Initial grid state with dirty cells marked as 'D'
- Optimal starting position
- Cleaning sequence
- Step-by-step movement and cleaning actions
- Final completion status

## Project Structure

```
vacuum-ai/
├── README.md
├── requirements.txt
├── vacum_ai.py
└── .gitignore
```

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- A* algorithm implementation inspired by various pathfinding resources
- TSP optimization based on standard algorithms
- Grid visualization techniques from similar projects 