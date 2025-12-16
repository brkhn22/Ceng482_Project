import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from game import SnakeGameAI, Direction, Point, BLOCK_SIZE
from multiprocessing import Pool
import os
import warnings
from collections import deque 

warnings.filterwarnings('ignore', category=RuntimeWarning)

# --- GENETIC ALGORITHM PARAMETERS ---
NUM_PROCESSES = max(1, os.cpu_count() - 1)
POPULATION_SIZE = 400         
MUTATION_RATE = 0.07          # Lowered slightly to preserve "smart" behaviors
MUTATION_STRENGTH = 0.2       
MAX_GENERATIONS = 200          

# === ARCHITECTURE CHANGES ===
INPUT_SIZE = 19               # CHANGED: 14 -> 19 (Flood Fill + Body Size + Tail)
HIDDEN_SIZE = 128             # Keep 128 for brain capacity
OUTPUT_SIZE = 3

# ----------------------------------------------------------
# NEURAL NETWORK
# ----------------------------------------------------------

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.weights1 = np.random.uniform(-1, 1, (input_size, hidden_size))
        self.bias1 = np.random.uniform(-1, 1, (1, hidden_size))
        
        self.weights2 = np.random.uniform(-1, 1, (hidden_size, output_size))
        self.bias2 = np.random.uniform(-1, 1, (1, output_size))

    def relu(self, x):
        return np.maximum(0, x)
    
    def predict(self, input_data):
        input_data = input_data.reshape(1, -1)
        hidden = self.relu(np.dot(input_data, self.weights1) + self.bias1)
        output = np.dot(hidden, self.weights2) + self.bias2
        action = [0,0,0]
        action[np.argmax(output)] = 1
        return action
    
    def mutate(self, mutation_rate=MUTATION_RATE, mutation_strength=MUTATION_STRENGTH):
        def mutate_matrix(matrix):
            mask = np.random.random(matrix.shape) < mutation_rate
            matrix[mask] += np.random.randn(*matrix.shape)[mask] * mutation_strength

        mutate_matrix(self.weights1)
        mutate_matrix(self.bias1)
        mutate_matrix(self.weights2)
        mutate_matrix(self.bias2)
    
    def copy(self):
        nn = NeuralNetwork(self.input_size, self.hidden_size, self.output_size)
        nn.weights1 = np.copy(self.weights1)
        nn.bias1 = np.copy(self.bias1)
        nn.weights2 = np.copy(self.weights2)
        nn.bias2 = np.copy(self.bias2)
        return nn

    @staticmethod
    def crossover(p1, p2):
        child = NeuralNetwork(p1.input_size, p1.hidden_size, p1.output_size)
        
        def mix(a, b):
            mask = np.random.rand(*a.shape) < 0.5
            return np.where(mask, a, b)
        
        child.weights1 = mix(p1.weights1, p2.weights1)
        child.bias1 = mix(p1.bias1, p2.bias1)
        child.weights2 = mix(p1.weights2, p2.weights2)
        child.bias2 = mix(p1.bias2, p2.bias2)
        return child

# ----------------------------------------------------------
# STATE FUNCTION (19 INPUTS: Flood + Size + Tail)
# ----------------------------------------------------------

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

    # --- 1. Flood Fill Logic ---
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
            neighbors = [
                (cx - BLOCK_SIZE, cy), (cx + BLOCK_SIZE, cy),
                (cx, cy - BLOCK_SIZE), (cx, cy + BLOCK_SIZE)
            ]
            for nx, ny in neighbors:
                if (0 <= nx < game.w and 0 <= ny < game.h and 
                    (nx, ny) not in occupied and (nx, ny) not in visited):
                    visited.add((nx, ny))
                    queue.append((nx, ny))
        return count / MAX_SEARCH

    # Relative Directions
    if dir_r:
        p_straight, p_right, p_left = point_r, point_d, point_u
    elif dir_l:
        p_straight, p_right, p_left = point_l, point_u, point_d
    elif dir_u:
        p_straight, p_right, p_left = point_u, point_r, point_l
    else: # dir_d
        p_straight, p_right, p_left = point_d, point_l, point_r
        
    # --- New Inputs Calculations ---
    total_grid_size = (game.w // BLOCK_SIZE) * (game.h // BLOCK_SIZE)
    norm_length = len(game.snake) / total_grid_size

    # --- 2. Construct State Vector ---
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
        get_accessible_area(p_straight),
        get_accessible_area(p_right),
        get_accessible_area(p_left),
        
        # NEW: Body Size (1)
        norm_length,

        # NEW: Tail Direction (4)
        tail.x < head.x, # Tail Left
        tail.x > head.x, # Tail Right
        tail.y < head.y, # Tail Up
        tail.y > head.y  # Tail Down
    ]

    return np.array(state, dtype=float)

# ----------------------------------------------------------
# FITNESS
# ----------------------------------------------------------

def evaluate_agent(nn, render=False, display_speed_multiplier=1):
    game = SnakeGameAI(w=640, h=480, render_mode=render, display_speed_multiplier=display_speed_multiplier)
    game.reset()

    steps = 0
    max_steps = 150 # Increased

    while True:
        state = get_state(game)
        action = nn.predict(state)
        reward, game_over, score = game.play_step(action)
        steps += 1
        
        if score > 0:
            max_steps = 150 + (score * 100)

        if game_over or steps > max_steps:
            break

    # CHANGED: Replaced 2**score with score**3 to prevent explosion
    fitness = (score ** 3) * 100 + steps + (score * 500)
    return max(0.1, fitness), score

def evaluate_agent_wrapper(nn):
    return evaluate_agent(nn, render=False)

# ----------------------------------------------------------
# GA OPERATIONS
# ----------------------------------------------------------

def select_best(population, fitnesses):
    order = np.argsort(fitnesses)[::-1]
    return [population[i] for i in order]

def next_generation(best_agents, population_size, all_time_best):
    new = []

    # --- 1. Absolute Elitism ---
    # Ensure the All-Time Best is always preserved (2 copies)
    if all_time_best:
        new.append(all_time_best.copy())
        new.append(all_time_best.copy())

    # --- 2. Current Generation Elitism ---
    # Keep top 3 from current generation
    for i in range(3):
        if len(new) < population_size:
            new.append(best_agents[i].copy())

    # --- 3. Crossover & Mutation ---
    # Parents come from the top 20%
    parent_pool_size = max(2, int(population_size * 0.2))
    parent_pool = best_agents[:parent_pool_size]
    
    # Add all-time best to the breeding pool as well
    if all_time_best:
        parent_pool.append(all_time_best)

    while len(new) < population_size:
        p1 = np.random.choice(parent_pool)
        p2 = np.random.choice(parent_pool)
        
        child = NeuralNetwork.crossover(p1, p2)
        child.mutate()
        new.append(child)

    return new

# ----------------------------------------------------------
# MAIN LOOP
# ----------------------------------------------------------

def main():
    print(f"Starting Genetic Algorithm with {NUM_PROCESSES} parallel processes...")
    print(f"Population: {POPULATION_SIZE}, Generations: {MAX_GENERATIONS}")
    print(f"Structure: {INPUT_SIZE} -> {HIDDEN_SIZE} -> {OUTPUT_SIZE}")
    print("-" * 60)
    
    population = [NeuralNetwork(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE) for _ in range(POPULATION_SIZE)]
    
    best_scores = []
    avg_scores = []
    gens = []
    
    all_time_best = None
    all_time_best_score = -1

    for gen in range(1, MAX_GENERATIONS + 1):
        with Pool(processes=NUM_PROCESSES) as pool:
            results = pool.map(evaluate_agent_wrapper, population)
        
        fitnesses = np.array([r[0] for r in results])
        scores = np.array([r[1] for r in results])

        current_best_score = np.max(scores)
        avg_score = np.mean(scores)
        
        best_agent_idx = np.argmax(scores)
        current_best_agent = population[best_agent_idx]

        # Update All-Time Best
        if current_best_score > all_time_best_score:
            all_time_best_score = current_best_score
            all_time_best = current_best_agent.copy()
            print(f"\n>>> NEW RECORD: {all_time_best_score} <<<")

        best_scores.append(current_best_score)
        avg_scores.append(avg_score)
        gens.append(gen)

        print(f"Gen {gen:3d} | Best: {current_best_score:3d} (Record: {all_time_best_score}) | Avg: {avg_score:.2f}")

        # Evolution step
        sorted_pop = select_best(population, fitnesses)
        if gen < MAX_GENERATIONS:
            population = next_generation(sorted_pop, POPULATION_SIZE, all_time_best)

    # Plot
    plt.figure(figsize=(10,6))
    plt.plot(gens, best_scores, label="Best (Current Gen)")
    plt.plot(gens, avg_scores, label="Average")
    plt.title("Genetic Algorithm Training (Flood+Tail)")
    plt.xlabel("Generation")
    plt.ylabel("Score")
    plt.legend()
    plt.savefig("ga_training_v3.png")

    print(f"\nTraining Complete! All-time Best: {all_time_best_score}")

    while True:
        cmd = input("\nWatch (b)est all time, or (q)uit? ").lower()
        if cmd == 'q': break
        if cmd == 'b' and all_time_best:
            evaluate_agent(all_time_best, render=True, display_speed_multiplier=5)

if __name__ == "__main__":
    main()