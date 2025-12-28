import numpy as np
import matplotlib
matplotlib.use('Agg') # Non-GUI backend for saving files
import matplotlib.pyplot as plt
from game import SnakeGameAI, Direction, Point, BLOCK_SIZE
from multiprocessing import Pool
import os
import warnings
from collections import deque 
import sys

warnings.filterwarnings('ignore', category=RuntimeWarning)

# ==========================================
# CONFIGURATION
# ==========================================
NUM_PROCESSES = max(1, os.cpu_count() - 1)

# Run Parameters
ITERATIONS = 2             # Number of separate training sessions
POPULATION_SIZE = 500          
MAX_GENERATIONS = 500       # Generations per iteration
ELITISM_SELECTION_PERCENT = 0.024

# Evolution Strategy Hyperparameters
MUTATION_RATE = 0.05        
MUTATION_STRENGTH = 0.2  
CROSSOVER_RATE = 0.4        

# Architecture
INPUT_SIZE = 24             
HIDDEN_SIZE = 16
OUTPUT_SIZE = 3        

# ----------------------------------------------------------
# 1. NEURAL NETWORK
# ----------------------------------------------------------
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # He Initialization
        self.weights1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0/input_size)
        self.bias1 = np.random.randn(1, hidden_size) * 0.1  
        self.weights2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0/hidden_size)
        self.bias2 = np.random.randn(1, output_size) * 0.1

    def relu(self, x):
        return np.maximum(0, x)
    
    def predict(self, input_data):
        input_data = input_data.reshape(1, -1)
        hidden = self.relu(np.dot(input_data, self.weights1) + self.bias1)
        output = np.dot(hidden, self.weights2) + self.bias2
        action = [0,0,0]
        action[np.argmax(output)] = 1
        return action
    
    def mutate(self):
        def mutate_matrix(matrix):
            mask = np.random.random(matrix.shape) < MUTATION_RATE
            noise = np.random.randn(*matrix.shape) * MUTATION_STRENGTH
            matrix[mask] += noise[mask]
            np.clip(matrix, -5.0, 5.0, out=matrix)
        
        mutate_matrix(self.weights1)
        mutate_matrix(self.bias1)
        mutate_matrix(self.weights2)
        mutate_matrix(self.bias2)

    @staticmethod
    def crossover(p1, p2):
        child = NeuralNetwork(p1.input_size, p1.hidden_size, p1.output_size)
        
        def select_genes(w1, w2):
            mask = np.random.rand(*w1.shape) < 0.5
            return np.where(mask, w1, w2)
        
        child.weights1 = select_genes(p1.weights1, p2.weights1)
        child.bias1 = select_genes(p1.bias1, p2.bias1)
        child.weights2 = select_genes(p1.weights2, p2.weights2)
        child.bias2 = select_genes(p1.bias2, p2.bias2)
        
        return child
        
    def copy(self):
        nn = NeuralNetwork(self.input_size, self.hidden_size, self.output_size)
        nn.weights1 = self.weights1.copy()
        nn.bias1 = self.bias1.copy()
        nn.weights2 = self.weights2.copy()
        nn.bias2 = self.bias2.copy()
        return nn


# ----------------------------------------------------------
# HELPER: FLOOD FILL
# ----------------------------------------------------------
def calculate_flood_fill(game):
    if len(game.snake) < 10: return 1.0
    head = game.snake[0]
    board_area = (game.w // BLOCK_SIZE) * (game.h // BLOCK_SIZE)
    obstacles = set((p.x, p.y) for p in game.snake)
    queue = deque([(head.x, head.y)])
    visited = set([(head.x, head.y)])
    count = 0
    limit = board_area * 0.5 
    moves = [(0, -BLOCK_SIZE), (0, BLOCK_SIZE), (-BLOCK_SIZE, 0), (BLOCK_SIZE, 0)]
    while queue:
        cx, cy = queue.popleft()
        count += 1
        if count > limit: return 1.0
        for dx, dy in moves:
            nx, ny = cx + dx, cy + dy
            if nx < 0 or nx >= game.w or ny < 0 or ny >= game.h: continue
            if (nx, ny) in obstacles: continue
            if (nx, ny) not in visited:
                visited.add((nx, ny))
                queue.append((nx, ny))
    return count / board_area

# ----------------------------------------------------------
# STATE FUNCTION
# ----------------------------------------------------------
def get_state(game):
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

# ----------------------------------------------------------
# FITNESS FUNCTION
# ----------------------------------------------------------
def evaluate_agent(nn):
    game = SnakeGameAI(w=640, h=480, render_mode=False) 
    game.reset()
    steps_total = 0
    steps_since_eaten = 0
    starvation_limit = 100 * len(game.snake) 
    
    while True:
        state = get_state(game)
        action = nn.predict(state)
        reward, game_over, score = game.play_step(action)
        steps_total += 1
        steps_since_eaten += 1
        
        if steps_since_eaten > starvation_limit:
            game_over = True
            reward = -10 
        if reward > 0:
            steps_since_eaten = 0
            starvation_limit = 200 # FIXED: Corrected starvation logic
        if game_over: break

        fitness_bonus = 0
      #  if score>50:
      #      flood_fill_val = calculate_flood_fill(game)
      #      fitness_bonus = (score ** 3) * (flood_fill_val * 0.1)

    return (score ** 3) + (steps_total * 0.001), score #+ fitness_bonus, score

def eval_wrapper(nn): return evaluate_agent(nn)

# ----------------------------------------------------------
# PLOTTING FUNCTIONS
# ----------------------------------------------------------
def plot_iteration_results(iteration_id, gen_history, best_history, avg_history):
    """Generates a line graph for a single iteration."""
    plt.figure(figsize=(10, 6))
    plt.plot(gen_history, best_history, label='Best Score', color='#1f77b4', linewidth=2)
    plt.plot(gen_history, avg_history, label='Avg Score', color='#ff7f0e', linestyle='--', alpha=0.7)
    
    plt.title(f'Iteration {iteration_id}: Training Progress')
    plt.xlabel('Generations')
    plt.ylabel('Score')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    
    filename = f"iteration_{iteration_id}.png"
    plt.savefig(filename)
    plt.close()
    print(f"   [Graph Saved]: {filename}")

def plot_final_summary(champions):
    """Generates a bar chart comparing best scores across all iterations."""
    ids = [c['id'] for c in champions]
    scores = [c['score'] for c in champions]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(ids, scores, color='#2ca02c', alpha=0.8, edgecolor='black')
    
    plt.title('Final Champion Scores per Iteration')
    plt.xlabel('Iteration ID')
    plt.ylabel('Best Score Achieved')
    plt.xticks(ids) # Ensure only integer IDs are shown
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Add score labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{int(height)}',
                ha='center', va='bottom')
    
    plt.savefig("final_summary_scores.png")
    plt.close()
    print(f"   [Summary Graph Saved]: final_summary_scores.png")

# ----------------------------------------------------------
# TRAINING LOOP FOR ONE ITERATION
# ----------------------------------------------------------
def train_iteration(iteration_id):
    print(f"\n" + "="*50)
    print(f"STARTING ITERATION {iteration_id} / {ITERATIONS}")
    print("="*50)
    
    population = [NeuralNetwork(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE) for _ in range(POPULATION_SIZE)]
    
    all_time_best_score = -1
    all_time_best_agent = None
    
    # Data Collection for Graphing
    gen_history = []
    best_history = []
    avg_history = []
    
    for gen in range(1, MAX_GENERATIONS + 1):
        with Pool(NUM_PROCESSES) as pool:
            results = pool.map(eval_wrapper, population)
        
        scores = np.array([r[1] for r in results])
        fitnesses = np.array([r[0] for r in results])
        
        best_gen_score = np.max(scores)
        avg_gen_score = np.mean(scores)
        
        # Save Stats
        gen_history.append(gen)
        best_history.append(best_gen_score)
        avg_history.append(avg_gen_score)
        
        if best_gen_score > all_time_best_score:
            all_time_best_score = best_gen_score
            best_idx = np.argmax(scores)
            all_time_best_agent = population[best_idx].copy()
            print(f"Iter {iteration_id} | Gen {gen:3d} | >>> NEW RECORD: {all_time_best_score}")
        
        if gen % 10 == 0:
            print(f"Iter {iteration_id} | Gen {gen:3d} | Best: {best_gen_score:3d} | Avg: {avg_gen_score:.2f}")

        # BREEDING
        new_pop = []
        indices = np.argsort(fitnesses)[::-1]

        # ELITISM
        num_random_max = int(POPULATION_SIZE * ELITISM_SELECTION_PERCENT)
        for i in range(num_random_max):
            parent_max = population[indices[i]].copy()
            new_pop.append(parent_max)
            child_max = parent_max.copy()
            child_max.mutate()
            new_pop.append(child_max)
            
        all_indices = np.arange(POPULATION_SIZE)
        while len(new_pop) < POPULATION_SIZE:
            t1_indices = np.random.choice(all_indices, size=4, replace=False)
            t1_winner_idx = t1_indices[np.argmax(fitnesses[t1_indices])]
            p1 = population[t1_winner_idx]
            
            if np.random.rand() < CROSSOVER_RATE:
                t2_indices = np.random.choice(all_indices, size=4, replace=False)
                t2_winner_idx = t2_indices[np.argmax(fitnesses[t2_indices])]
                p2 = population[t2_winner_idx]
                child = NeuralNetwork.crossover(p1, p2)
            else:
                child = p1.copy()
            
            child.mutate()
            new_pop.append(child)
            
        population = new_pop

    print(f"-> Iteration {iteration_id} Finished. Best Score: {all_time_best_score}")
    
    # 
    # Generate graph for this iteration
    plot_iteration_results(iteration_id, gen_history, best_history, avg_history)
    
    return all_time_best_agent, all_time_best_score

# ----------------------------------------------------------
# MAIN EXECUTION
# ----------------------------------------------------------
if __name__ == "__main__":
    
    champions = [] # Stores dictionary: {'id': int, 'agent': nn, 'score': int}
    
    # 1. RUN ITERATIONS
    for i in range(1, ITERATIONS + 1):
        best_agent, best_score = train_iteration(i)
        champions.append({
            'id': i,
            'agent': best_agent,
            'score': best_score
        })
    
    # 2. GENERATE FINAL SUMMARY GRAPH
    # 
    plot_final_summary(champions)

    # 3. CALCULATE STATS
    scores = [c['score'] for c in champions]
    min_best = min(scores)
    max_best = max(scores)
    avg_best = sum(scores) / len(scores)
    
    print("\n" + "#"*60)
    print("ALL TRAINING COMPLETE")
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
            
            # Find the champion
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