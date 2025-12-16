import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from game import SnakeGameAI, Direction, Point, BLOCK_SIZE
from multiprocessing import Pool
import os
import time
import warnings
from collections import deque 

warnings.filterwarnings('ignore', category=RuntimeWarning)

# === OPTIMIZED HYPERPARAMETERS ===
NUM_PROCESSES = max(1, os.cpu_count() - 1)
MU = 50                     
LAMBDA = 450                
MAX_GENERATIONS = 200

# === ARCHITECTURE ===
INPUT_SIZE = 19             # Flood Fill + Tail + Length
HIDDEN_SIZE = 128           
OUTPUT_SIZE = 3

# Mutation parameters
INITIAL_SIGMA = 0.5         
MIN_SIGMA = 0.05            
MAX_SIGMA = 1.0             
TAU = 0.05                  

class ESNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.weights1 = np.random.uniform(-1, 1, (input_size, hidden_size))
        self.bias1 = np.random.uniform(-1, 1, (1, hidden_size))
        self.weights2 = np.random.uniform(-1, 1, (hidden_size, output_size))
        self.bias2 = np.random.uniform(-1, 1, (1, output_size))
        
        self.sigma = INITIAL_SIGMA

    def relu(self, x):
        return np.maximum(0, x)

    def predict(self, input_data):
        if len(input_data.shape) == 1:
            input_data = input_data.reshape(1, -1)
        hidden = self.relu(np.dot(input_data, self.weights1) + self.bias1)
        output = np.dot(hidden, self.weights2) + self.bias2
        action = [0, 0, 0]
        action[np.argmax(output)] = 1
        return action

    def es_mutate(self):
        # Self-adaptive sigma
        self.sigma *= np.exp(TAU * np.random.randn())
        self.sigma = max(MIN_SIGMA, min(self.sigma, MAX_SIGMA))

        def mutate(mat):
            return mat + np.random.randn(*mat.shape) * self.sigma

        self.weights1 = mutate(self.weights1)
        self.bias1 = mutate(self.bias1)
        self.weights2 = mutate(self.weights2)
        self.bias2 = mutate(self.bias2)

    def copy(self):
        nn = ESNeuralNetwork(self.input_size, self.hidden_size, self.output_size)
        nn.weights1 = np.copy(self.weights1)
        nn.bias1 = np.copy(self.bias1)
        nn.weights2 = np.copy(self.weights2)
        nn.bias2 = np.copy(self.bias2)
        nn.sigma = self.sigma
        return nn

# === STATE FUNCTION (19 Inputs) ===
def get_state(game):
    head = game.snake[0]
    tail = game.snake[-1]
    
    point_l = Point(head.x - BLOCK_SIZE, head.y)
    point_r = Point(head.x + BLOCK_SIZE, head.y)
    point_u = Point(head.x, head.y - BLOCK_SIZE)
    point_d = Point(head.x, head.y + BLOCK_SIZE)

    dir_l = game.direction == Direction.LEFT
    dir_r = game.direction == Direction.RIGHT
    dir_u = game.direction == Direction.UP
    dir_d = game.direction == Direction.DOWN

    occupied = set((p.x, p.y) for p in game.snake)
    
    # Flood Fill
    def get_accessible_area(start_point):
        if (start_point.x < 0 or start_point.x >= game.w or 
            start_point.y < 0 or start_point.y >= game.h or 
            (start_point.x, start_point.y) in occupied):
            return 0.0
        
        visited = set()
        queue = deque([(start_point.x, start_point.y)])
        visited.add((start_point.x, start_point.y))
        count = 0
        MAX_SEARCH = 80
        while queue and count < MAX_SEARCH:
            cx, cy = queue.popleft()
            count += 1
            for nx, ny in [(cx-BLOCK_SIZE, cy), (cx+BLOCK_SIZE, cy), 
                           (cx, cy-BLOCK_SIZE), (cx, cy+BLOCK_SIZE)]:
                if (0 <= nx < game.w and 0 <= ny < game.h and 
                    (nx, ny) not in occupied and (nx, ny) not in visited):
                    visited.add((nx, ny))
                    queue.append((nx, ny))
        return count / MAX_SEARCH

    if dir_r: p_s, p_r, p_l = point_r, point_d, point_u
    elif dir_l: p_s, p_r, p_l = point_l, point_u, point_d
    elif dir_u: p_s, p_r, p_l = point_u, point_r, point_l
    else: p_s, p_r, p_l = point_d, point_l, point_r

    total_grid_size = (game.w // BLOCK_SIZE) * (game.h // BLOCK_SIZE)
    norm_length = len(game.snake) / total_grid_size

    state = [
        # Danger (3)
        (dir_r and game.is_collision(point_r)) or (dir_l and game.is_collision(point_l)) or (dir_u and game.is_collision(point_u)) or (dir_d and game.is_collision(point_d)),
        (dir_u and game.is_collision(point_r)) or (dir_d and game.is_collision(point_l)) or (dir_l and game.is_collision(point_u)) or (dir_r and game.is_collision(point_d)),
        (dir_d and game.is_collision(point_r)) or (dir_u and game.is_collision(point_l)) or (dir_r and game.is_collision(point_u)) or (dir_l and game.is_collision(point_d)),
        # Direction (4)
        dir_l, dir_r, dir_u, dir_d,
        # Food (4)
        game.food.x < head.x, game.food.x > head.x,
        game.food.y < head.y, game.food.y > head.y,
        # Flood Fill (3)
        get_accessible_area(p_s), get_accessible_area(p_r), get_accessible_area(p_l),
        # Body Size (1)
        norm_length,
        # Tail (4)
        tail.x < head.x, tail.x > head.x, tail.y < head.y, tail.y > head.y
    ]
    return np.array(state, dtype=float)


# === FIXED FITNESS FUNCTION ===
def evaluate_agent(nn, render=False, display_speed_multiplier=1):
    game = SnakeGameAI(w=640, h=480, render_mode=render, display_speed_multiplier=display_speed_multiplier)
    game.reset()

    steps = 0
    max_steps = 150 

    while True:
        state = get_state(game)
        action = nn.predict(state)
        reward, game_over, score = game.play_step(action)

        steps += 1
        if score > 0:
            max_steps = 150 + (score * 100) 
        
        if game_over or steps > max_steps:
            break

    # Reverted to Cubic fitness (Matches your successful GA)
    # 2**score is too unstable for ES
    fitness = (score ** 3) * 100 + steps + (score * 500)
    
    return max(0.1, fitness), score

def evaluate_agent_wrapper(nn):
    return evaluate_agent(nn, render=False)


# === TRAINING LOOP (MU + LAMBDA) ===
def main():
    print("\n=== STARTING ES (MU + LAMBDA) STRATEGY ===")
    print("Survival of the fittest (Parents compete with children)")
    print(f"Processes: {NUM_PROCESSES}")
    print(f"Generations: {MAX_GENERATIONS}")
    print("-" * 60)

    # 1. Initialize Parents
    parents_list = [ESNeuralNetwork(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE) for _ in range(MU)]
    
    # We must evaluate initial parents to get their fitness
    with Pool(processes=NUM_PROCESSES) as pool:
        results = pool.map(evaluate_agent_wrapper, parents_list)
    
    # Store population as dictionaries to avoid re-evaluating parents
    population_data = []
    for i, agent in enumerate(parents_list):
        population_data.append({
            'agent': agent,
            'fitness': results[i][0],
            'score': results[i][1]
        })

    best_scores = []
    avg_scores = []
    gen_idx = []
    
    all_time_best_score = -1
    stagnation = 0

    for gen in range(MAX_GENERATIONS):
        
        # 2. Create Offspring (Mutate copies of current population)
        offspring_agents = []
        for _ in range(LAMBDA):
            # Pick a parent at random from the survivors
            parent_data = np.random.choice(population_data)
            parent_agent = parent_data['agent']
            
            child = parent_agent.copy()
            child.es_mutate()
            offspring_agents.append(child)

        # 3. Evaluate ONLY Offspring
        with Pool(processes=NUM_PROCESSES) as pool:
            results = pool.map(evaluate_agent_wrapper, offspring_agents)
        
        offspring_data = []
        for i, agent in enumerate(offspring_agents):
            offspring_data.append({
                'agent': agent,
                'fitness': results[i][0],
                'score': results[i][1]
            })

        # 4. (MU + LAMBDA) SELECTION
        # Combine old parents and new children
        combined_population = population_data + offspring_data
        
        # Sort by fitness (High to Low)
        combined_population.sort(key=lambda x: x['fitness'], reverse=True)
        
        # Survivor Selection: Top MU agents survive
        population_data = combined_population[:MU]
        
        # 5. Stats
        current_best = population_data[0]
        current_best_score = current_best['score']
        avg_score = sum(p['score'] for p in population_data) / MU

        if current_best_score > all_time_best_score:
            all_time_best_score = current_best_score
            stagnation = 0
            print(f"\n>>> NEW RECORD: {all_time_best_score} <<<")
        else:
            stagnation += 1

       # === REPLACEMENT START ===
        # OLD: Just boosted sigma
        # NEW: MASS EXTINCTION EVENT
        if stagnation > 15:  # Trigger faster (15 gens instead of 20)
            print(f"\n!!! STAGNATION DETECTED (Best stuck at {all_time_best_score}) !!!")
            print(">>> INITIATING MASS EXTINCTION EVENT <<<")
            
            # 1. ELITISM: Keep only the Top 5 Agents (The "Kings")
            # We already sorted population_data by fitness, so just slice the top 5
            survivors = population_data[:5]
            
            # 2. EXTINCTION: Create completely NEW random agents to fill the rest
            num_new_agents = MU - 5
            print(f">>> Keeping 5 Elites, Injecting {num_new_agents} Fresh Random Agents...")
            
            new_blood = [ESNeuralNetwork(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE) for _ in range(num_new_agents)]
            
            # 3. Evaluate the new agents immediately
            with Pool(processes=NUM_PROCESSES) as pool:
                results = pool.map(evaluate_agent_wrapper, new_blood)
            
            # 4. Add them to the population list
            for i, agent in enumerate(new_blood):
                # Give new agents a high sigma so they explore aggressively
                agent.sigma = 0.8 
                survivors.append({
                    'agent': agent,
                    'fitness': results[i][0],
                    'score': results[i][1]
                })
            
            # 5. Overwrite the population with our new mixed group
            population_data = survivors
            stagnation = 0 # Reset counter
        # === REPLACEMENT END ===
        
        best_scores.append(current_best_score)
        avg_scores.append(avg_score)
        gen_idx.append(gen + 1)

        print(f"Gen {gen+1:3d} | Best: {current_best_score:3d} | Avg: {avg_score:6.2f} | Sigma: {population_data[0]['agent'].sigma:.3f}")

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(gen_idx, best_scores, label="Best")
    plt.plot(gen_idx, avg_scores, label="Avg", linestyle="--")
    plt.title("ES (Mu + Lambda) Training Progress")
    plt.xlabel("Generation")
    plt.ylabel("Score")
    plt.legend()
    plt.savefig("es_training_plus.png")
    plt.close()

    print("\n=== TRAINING COMPLETE ===")
    print(f"All-time best score: {all_time_best_score}")

    while True:
        cmd = input("\nWatch (b)est agent or (q)uit? ").lower()
        if cmd == 'q': break
        if cmd == 'b':
            evaluate_agent(population_data[0]['agent'], render=True, display_speed_multiplier=5)

if __name__ == "__main__":
    main()