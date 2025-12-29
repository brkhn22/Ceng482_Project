# Snake AI Project

This project implements two evolutionary algorithms to train neural networks to play the classic Snake game:

- **Genetic Algorithm (GA)**
- **Evolution Strategy (ES)**

## Contents
- `genetic_algorithm_main.py` — Genetic Algorithm for Snake AI
- `evolution_strategy_main.py` — Evolution Strategy for Snake AI
- `game.py` — Snake game environment (with AI interface)
- `brain.py` — (if present) Alternative neural network logic
- `requirements.txt` — Python dependencies

---

## 1. Snake Game

The game is a grid-based Snake implementation with:
- Customizable board size
- Food spawning
- Collision detection (walls, self)
- AI interface for step-by-step control

The AI interacts with the game by receiving a state vector and returning an action (move direction).

---

## 2. Genetic Algorithm (GA)

- **File:** `genetic_algorithm_main.py`
- **Approach:**
  - Population of neural networks
  - Each network plays the game, fitness = score + survival
  - Selection: Top performers are chosen
  - Crossover: Mix weights of two parents
  - Mutation: Add random noise to weights
  - Elitism: Best agents are preserved
- **State Representation:**
  - Ray-based vision (distance to wall, food, body in 8 directions)
  - Normalized inputs for neural network
- **Usage:**
  ```bash
  python genetic_algorithm_main.py
  ```
  - Trains for multiple generations, shows best agents

---

## 3. Evolution Strategy (ES)

- **File:** `evolution_strategy_main.py`
- **Approach:**
  - Uses (μ + λ) strategy: μ parents, λ offspring
  - Each agent has adaptive mutation strength (sigma)
  - Sparse mutation: Only a fraction of weights mutated per generation
  - Selection: Best μ agents survive
  - No crossover (mutation only)
- **State Representation:**
  - Same ray-based vision as GA
- **Usage:**
  ```bash
  python evolution_strategy_main.py
  ```
  - Trains for multiple generations, shows best agents

---

## 4. Requirements

- Python 3.8+
- `numpy`, `matplotlib`, `pygame`

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## 5. How to Run

1. Install requirements
2. Run either algorithm script
3. Follow on-screen instructions to watch trained agents

---

## 6. Visualization

- Training progress is saved as PNG graphs per iteration
- Final summary bar chart compares best scores across runs

---

## 7. Customization

- Change hyperparameters at the top of each script
- Modify neural network architecture as needed
- Adjust state representation for different strategies

---

## 8. Credits

Developed for CENG482 Project — Evolutionary AI for Snake Game

---