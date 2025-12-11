# Snake AI - Genetic Algorithm Training

This project implements a Snake game AI using Neuroevolution (Genetic Algorithm) with a neural network.

## Files

- **game.py**: Snake game environment optimized for AI training
- **brain.py**: Neural network implementation using NumPy
- **main.py**: Genetic algorithm training loop
- **requirements.txt**: Python dependencies

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the training:

```bash
python main.py
```

## How It Works

### 1. Snake Game (game.py)
- **Render Mode**: Optional `render_mode` parameter (default: True)
- **Speed**: When `render_mode=False`, runs at maximum CPU speed
- **Display**: Pygame window only created when rendering is enabled

### 2. Neural Network (brain.py)
- **Input**: 11 nodes representing snake vision (danger, direction, food location)
- **Hidden Layer**: 256 nodes with ReLU activation
- **Output**: 3 nodes (straight, right turn, left turn)

### 3. Genetic Algorithm (main.py)
- **Population**: 50 neural networks
- **Fitness**: `(score × 1000) + steps_alive - (turns × 5)`
  - Eating food is heavily prioritized (×1000)
  - Survival time rewarded (+steps_alive)
  - Excessive turning slightly penalized (-turns × 5)
  - Result: Snakes evolve to eat food while moving naturally in straighter lines
- **Selection**: Top 10% agents survive
- **Mutation**: 5% chance per weight
- **Timeout**: Snake dies if it doesn't eat within `100 × snake_length` frames

### 4. Evolution Process
1. Create random population of neural networks
2. Each agent plays the game until death
3. Calculate fitness for each agent
4. Select best performers (top 10%)
5. Create next generation by copying and mutating best agents
6. Repeat for 100 generations

## State Representation (11 values)

The neural network receives 11 boolean inputs:
- **Danger** (3): Straight, Right, Left relative to current direction
- **Direction** (4): Moving Left, Right, Up, Down
- **Food** (4): Food is Left, Right, Up, Down relative to head

## Anti-Spinning & Natural Movement

To prevent agents from exploiting the fitness function and encourage natural movement:

1. **Frame Timeout**: Snake must eat food within `100 × len(snake)` frames, or it dies
2. **Counter Reset**: The frame counter resets to 0 every time food is eaten
3. **Food-Focused Fitness**: Eating 1 food (1000 points) > surviving 999 frames (999 points)
4. **Turn Penalty**: Each direction change reduces fitness by 5 points
   - Encourages straighter, more natural movement patterns
   - Prevents excessive zigzagging
   - If two snakes have the same score, the one moving in straighter lines wins

This ensures agents evolve to seek food efficiently with natural movement!

## Results

After training completes:
- Console output shows progress per generation
- `training_progress.png` saved with evolution plots
- Best score and average score tracked over generations

## Parameters

You can modify these in `main.py`:
- `POPULATION_SIZE = 50`
- `MUTATION_RATE = 0.05`
- `MAX_GENERATIONS = 100`
- `HIDDEN_SIZE = 256`

## Performance Optimization

**Rendering is disabled during training** for maximum speed:
- Game logic runs as fast as CPU allows (no frame limits)
- No Pygame drawing operations during evolution
- Training is 10-100x faster than with rendering enabled

After training completes, you'll be prompted to watch the best agent play with rendering enabled.

## Debug Features

When watching the best agent play (after training), death causes are shown:
- **💀 Died: COLLISION - Hit Wall** - Snake hit the boundary
- **💀 Died: COLLISION - Hit Self** - Snake hit its own body
- **⌛ Died: STARVATION** - Snake didn't eat food within the frame limit

**Note**: Death messages are hidden during training to keep console output clean. They only appear when you choose to watch the best 5 games after training completes.

## Tips

- Training now takes only 1-5 minutes depending on your hardware
- Rendering is disabled during training for maximum speed
- Console output is clean during training (no spam)
- After training, watch 5 games with the best agent to see death causes
- Increase `MAX_GENERATIONS` for better results
- Adjust `MUTATION_RATE` if evolution stagnates
- Set `render_mode=True, verbose=True` in `SnakeGameAI()` to watch agents play with debug info

