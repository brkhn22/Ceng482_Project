import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from game import SnakeGameAI, Direction, Point, BLOCK_SIZE
from multiprocessing import Pool
import os
import warnings
from collections import deque 
import sys

warnings.filterwarnings('ignore', category=RuntimeWarning)


# Multiprocessing, parameters and architecture settings
NUM_PROCESSES = max(1, os.cpu_count() - 1)
ITERATIONS = 5
MU = 20           # Main population size
LAMBDA = 400      # Number of offspring
MAX_GENERATIONS = 250
INPUT_SIZE = 24   # Input size (rays)
HIDDEN_SIZE = 24  # Hidden layer size
OUTPUT_SIZE = 3   # Output (directions)

# Mutation parameters
INITIAL_SIGMA = 0.3         # Initial mutation strength
MIN_SIGMA = 0.1             # Minimum sigma
MAX_SIGMA = 0.6             # Maximum sigma
TAU = 0.05                  # Sigma adaptation rate
SPARSITY = 0.05             # Mutation rate

class ESNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
    # Initialize network weights
        self.weights1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.bias1 = np.random.randn(1, hidden_size) * 0.01
        self.weights2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.bias2 = np.random.randn(1, output_size) * 0.01
        self.sigma = INITIAL_SIGMA

    def relu(self, x):
        return np.maximum(0, x)

    def predict(self, input_data):
    # Pass input vector through network, select highest output
        if len(input_data.shape) == 1:
            input_data = input_data.reshape(1, -1)
        hidden = self.relu(np.dot(input_data, self.weights1) + self.bias1)
        output = np.dot(hidden, self.weights2) + self.bias2
        action = [0, 0, 0]
        action[np.argmax(output)] = 1
        return action

    def es_mutate(self):
    # Update sigma adaptively
        self.sigma *= np.exp(TAU * np.random.randn())
        self.sigma = max(MIN_SIGMA, min(self.sigma, MAX_SIGMA))

    # Apply sparse mutation to network weights
        def mutate(mat):
            mask = np.random.rand(*mat.shape) < SPARSITY
            noise = np.random.randn(*mat.shape) * self.sigma
            mat[mask] += noise[mask]
            return mat

        self.weights1 = mutate(self.weights1)
        self.bias1 = mutate(self.bias1)
        self.weights2 = mutate(self.weights2)
        self.bias2 = mutate(self.bias2)

    def copy(self):
    # Copy the network
        nn = ESNeuralNetwork(self.input_size, self.hidden_size, self.output_size)
        nn.weights1 = np.copy(self.weights1)
        nn.bias1 = np.copy(self.bias1)
        nn.weights2 = np.copy(self.weights2)
        nn.bias2 = np.copy(self.bias2)
        nn.sigma = self.sigma
        return nn


def get_state(game):
    # For 8 ray directions: distance, food and body info
    head = game.snake[0]
    body_set = set((p.x, p.y) for p in game.snake[1:])
    standard_angles = [(0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)]
    if game.direction == Direction.UP: shift = 0
    elif game.direction == Direction.RIGHT: shift = 2
    elif game.direction == Direction.DOWN: shift = 4
    else: shift = 6 
    current_rays = standard_angles[shift:] + standard_angles[:shift]
    state = []
    for dx, dy in current_rays:
        cx, cy = head.x, head.y
        dist = 0
        found_food = 0
        found_body = 0
        step_x = dx * BLOCK_SIZE
        step_y = dy * BLOCK_SIZE
        while True:
            cx += step_x
            cy += step_y
            dist += 1
            if cx < 0 or cx >= game.w or cy < 0 or cy >= game.h: break 
            if (cx, cy) in body_set:
                found_body = 1
                break 
            if cx == game.food.x and cy == game.food.y: found_food = 1
        norm_dist = 1.0 / dist 
        state.extend([norm_dist, found_food, found_body])
    return np.array(state, dtype=float)

def evaluate_agent(nn):
    # Play agent, calculate fitness by score and survival time
    game = SnakeGameAI(w=640, h=480, render_mode=False) 
    game.reset()
    steps = 0
    steps_since_eaten = 0
    starvation_limit = 100 * len(game.snake) 
    fitness_bonus = 0
    while True:
        state = get_state(game)
        action = nn.predict(state)
        reward, game_over, score = game.play_step(action)
        steps += 1
        steps_since_eaten += 1
        if steps_since_eaten > starvation_limit:
            game_over = True
            reward = -10
        if reward > 0:
            steps_since_eaten = 0
            starvation_limit = 200
        if game_over: break
    fitness = (score ** 3) + (steps * 0.001)
    return max(0.1, fitness), score

def evaluate_agent_wrapper(nn): return evaluate_agent(nn)

# ----------------------------------------------------------
# PLOTTING FUNCTIONS
# ----------------------------------------------------------
def plot_iteration_results(iteration_id, gen_history, best_history, avg_history):
    """Generates a line graph for a single iteration."""
    plt.figure(figsize=(10, 6))
    plt.plot(gen_history, best_history, label='Best Score', color='#d62728', linewidth=2)
    plt.plot(gen_history, avg_history, label='Avg Score', color='#1f77b4', linestyle='--', alpha=0.7)
    
    plt.title(f'ES Iteration {iteration_id}: Training Progress')
    plt.xlabel('Generations')
    plt.ylabel('Score')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    
    filename = f"es_iteration_{iteration_id}.png"
    plt.savefig(filename)
    plt.close()
    print(f"   [Graph Saved]: {filename}")

def plot_final_summary(champions):
    """Generates a bar chart comparing best scores across all iterations."""
    ids = [c['id'] for c in champions]
    scores = [c['score'] for c in champions]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(ids, scores, color='#9467bd', alpha=0.8, edgecolor='black')
    
    plt.title('Final ES Champion Scores per Iteration')
    plt.xlabel('Iteration ID')
    plt.ylabel('Best Score Achieved')
    plt.xticks(ids)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Add score labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{int(height)}',
                ha='center', va='bottom')
    
    plt.savefig("es_final_summary_scores.png")
    plt.close()
    print(f"   [Summary Graph Saved]: es_final_summary_scores.png")

# ----------------------------------------------------------
# TRAINING LOOP FOR ONE ITERATION
# ----------------------------------------------------------
def train_iteration(iteration_id):
    print(f"\n" + "="*50)
    print(f"STARTING ES ITERATION {iteration_id} / {ITERATIONS}")
    print(f"Strategy: MU+LAMBDA | Min Sigma {MIN_SIGMA}")
    print("="*50)

    # Initialize main population
    parents_list = [ESNeuralNetwork(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE) for _ in range(MU)]

    # Evaluate initial population
    with Pool(processes=NUM_PROCESSES) as pool:
        results = pool.map(evaluate_agent_wrapper, parents_list)

    population_data = []
    for i, agent in enumerate(parents_list):
        population_data.append({'agent': agent, 'fitness': results[i][0], 'score': results[i][1]})

    all_time_best_score = -1
    all_time_best_agent = None
    stagnation = 0

    gen_history = []
    best_history = []
    avg_history = []

    for gen in range(1, MAX_GENERATIONS + 1):
    # Generate offspring
        offspring_agents = []
        for _ in range(LAMBDA):
            parent_data = np.random.choice(population_data)
            child = parent_data['agent'].copy()
            child.es_mutate()
            offspring_agents.append(child)

    # Evaluate offspring
        with Pool(processes=NUM_PROCESSES) as pool:
            results = pool.map(evaluate_agent_wrapper, offspring_agents)

        offspring_data = []
        for i, agent in enumerate(offspring_agents):
            offspring_data.append({'agent': agent, 'fitness': results[i][0], 'score': results[i][1]})

    # Select best MU agents
        combined = population_data + offspring_data
        combined.sort(key=lambda x: x['fitness'], reverse=True)
        population_data = combined[:MU]

    # Save statistics
        current_best = population_data[0]
        current_best_score = current_best['score']
        avg_gen_score = sum(p['score'] for p in population_data) / MU

        gen_history.append(gen)
        best_history.append(current_best_score)
        avg_history.append(avg_gen_score)

        if current_best_score > all_time_best_score:
            all_time_best_score = current_best_score
            all_time_best_agent = current_best['agent'].copy()
            stagnation = 0
            print(f"Iter {iteration_id} | Gen {gen:3d} | >>> NEW RECORD: {all_time_best_score}")
        else:
            stagnation += 1

    # Print status every 10 generations
        if gen % 10 == 0:
            print(f"Iter {iteration_id} | Gen {gen:3d} | Best: {current_best_score:3d} | Avg: {avg_gen_score:.2f} | Sigma: {population_data[0]['agent'].sigma:.3f}")

    print(f"-> Iteration {iteration_id} Finished. Best Score: {all_time_best_score}")

    plot_iteration_results(iteration_id, gen_history, best_history, avg_history)

    return all_time_best_agent, all_time_best_score

# ----------------------------------------------------------
# MAIN EXECUTION
# ----------------------------------------------------------
if __name__ == "__main__":
    
    champions = [] 
    
    # 1. RUN ITERATIONS
    for i in range(1, ITERATIONS + 1):
        best_agent, best_score = train_iteration(i)
        champions.append({
            'id': i,
            'agent': best_agent,
            'score': best_score
        })
    
    # 2. GENERATE FINAL SUMMARY GRAPH
    plot_final_summary(champions)

    # 3. CALCULATE STATS
    scores = [c['score'] for c in champions]
    min_best = min(scores)
    max_best = max(scores)
    avg_best = sum(scores) / len(scores)
    
    print("\n" + "#"*60)
    print("ALL ES TRAINING COMPLETE")
    print("#"*60)
    print(f"Total Iterations: {ITERATIONS}")
    print(f"Minimum Best Score: {min_best}")
    print(f"Maximum Best Score: {max_best}")
    print(f"Average Best Score: {avg_best:.2f}")
    
    print("\nPer Iteration Breakdown:")
    for c in champions:
        print(f"Iter {c['id']:2d}: Score {c['score']}")
    print("#"*60)
    
    # 4. UI SIMULATION LOOP
    while True:
        try:
            user_input = input(f"\nEnter Iteration ID (1-{ITERATIONS}) to watch, or 'q' to quit: ").strip().lower()
            if user_input == 'q':
                print("Exiting...")
                break
            
            iter_id = int(user_input)
            selected_champ = next((c for c in champions if c['id'] == iter_id), None)
            
            if selected_champ:
                print(f"Simulating Champion from Iteration {iter_id} (Score {selected_champ['score']})...")
                game = SnakeGameAI(w=640, h=480, render_mode=True, display_speed_multiplier=5)
                game.reset()
                agent = selected_champ['agent']
                
                while True:
                    state = get_state(game)
                    action = agent.predict(state)
                    _, done, score = game.play_step(action)
                    if done: 
                        print(f"Game Over. Final Score: {score}")
                        break
            else:
                print(f"Invalid ID. Please enter 1-{ITERATIONS}.")
        except ValueError:
            print("Invalid input.")
        except KeyboardInterrupt:
            print("\nForce Quit.")
            break