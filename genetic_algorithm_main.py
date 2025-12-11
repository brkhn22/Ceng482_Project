import numpy as np
import matplotlib.pyplot as plt
from game import SnakeGameAI, Direction, Point, BLOCK_SIZE

# --- GENETİK ALGORİTMA PARAMETRELERİ ---
POPULATION_SIZE = 300         
MUTATION_RATE = 0.08           
MUTATION_STRENGTH = 0.15       
MAX_GENERATIONS = 200          
INPUT_SIZE = 17                
HIDDEN_SIZE = 32               
OUTPUT_SIZE = 3

# ----------------------------------------------------------
# NÖRON AĞI
# ----------------------------------------------------------

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Xavier değil! GA için uniform daha stabil.
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
# STATE FONKSİYONU (16 INPUT)
# ----------------------------------------------------------

def get_state(game):
    head = game.snake[0]

    total_grid_size = (game.w // BLOCK_SIZE) * (game.h // BLOCK_SIZE)
    normalized_length = len(game.snake) / total_grid_size
    
    state.append(normalized_length)

    point_l = Point(head.x - BLOCK_SIZE, head.y)
    point_r = Point(head.x + BLOCK_SIZE, head.y)
    point_u = Point(head.x, head.y - BLOCK_SIZE)
    point_d = Point(head.x, head.y + BLOCK_SIZE)

    dir_l = game.direction == Direction.LEFT
    dir_r = game.direction == Direction.RIGHT
    dir_u = game.direction == Direction.UP
    dir_d = game.direction == Direction.DOWN

    # --- Tehlikeler ---
    if dir_r:
        danger_straight = game.is_collision(point_r)
        danger_right = game.is_collision(point_d)
        danger_left = game.is_collision(point_u)
    elif dir_l:
        danger_straight = game.is_collision(point_l)
        danger_right = game.is_collision(point_u)
        danger_left = game.is_collision(point_d)
    elif dir_u:
        danger_straight = game.is_collision(point_u)
        danger_right = game.is_collision(point_r)
        danger_left = game.is_collision(point_l)
    else: # dir_d
        danger_straight = game.is_collision(point_d)
        danger_right = game.is_collision(point_l)
        danger_left = game.is_collision(point_r)

    state = [
        int(danger_straight),
        int(danger_right),
        int(danger_left)
    ]

    # ---- Raycasting: 3 mesafe ---
    if dir_r: directions = [Point(BLOCK_SIZE,0), Point(0,BLOCK_SIZE), Point(0,-BLOCK_SIZE)]
    elif dir_l: directions = [Point(-BLOCK_SIZE,0), Point(0,-BLOCK_SIZE), Point(0,BLOCK_SIZE)]
    elif dir_u: directions = [Point(0,-BLOCK_SIZE), Point(BLOCK_SIZE,0), Point(-BLOCK_SIZE,0)]
    else: directions = [Point(0,BLOCK_SIZE), Point(-BLOCK_SIZE,0), Point(BLOCK_SIZE,0)]

    for d in directions:
        dist = 1
        cur = Point(head.x, head.y)
        while dist < 20:
            cur = Point(cur.x + d.x, cur.y + d.y)
            if game.is_collision(cur):
                break
            dist += 1
        state.append(1 / dist)

    # --- Yön ---
    state.extend([int(dir_l), int(dir_r), int(dir_u), int(dir_d)])

    # --- Yemek pozisyonu ---
    food = game.food
    state.extend([
        int(food.x < head.x),
        int(food.x > head.x),
        int(food.y < head.y),
        int(food.y > head.y),
    ])

    # --- Alan kontrolü (Space awareness) ---
    def measure_space(start, direction):
        space = 0
        cur = start
        for _ in range(5):
            cur = Point(cur.x + direction.x, cur.y + direction.y)
            if game.is_collision(cur): break
            space += 1
        return space / 5.0

    if dir_r: vec_left, vec_right = Point(0,-BLOCK_SIZE), Point(0,BLOCK_SIZE)
    elif dir_l: vec_left, vec_right = Point(0,BLOCK_SIZE), Point(0,-BLOCK_SIZE)
    elif dir_u: vec_left, vec_right = Point(-BLOCK_SIZE,0), Point(BLOCK_SIZE,0)
    else: vec_left, vec_right = Point(BLOCK_SIZE,0), Point(-BLOCK_SIZE,0)

    state.append(measure_space(head, vec_left))
    state.append(measure_space(head, vec_right))

    return np.array(state, dtype=float)

# ----------------------------------------------------------
# FITNESS (Optimize Edilmiş)
# ----------------------------------------------------------

def evaluate_agent(nn, render=False):
    game = SnakeGameAI(w=640, h=480, render_mode=render)
    game.reset()

    steps = 0
    max_steps = 120

    while True:
        state = get_state(game)
        action = nn.predict(state)

        reward, game_over, score = game.play_step(action)

        steps += 1
        max_steps = 120 + score * 40

        if game_over or steps > max_steps:
            break

    # ---- Fitness ----
    return score * 500 + steps, score

# ----------------------------------------------------------
# GA İŞLEMLERİ
# ----------------------------------------------------------

def select_best(population, fitnesses):
    order = np.argsort(fitnesses)[::-1]
    return [population[i] for i in order]

def next_generation(best, population_size):
    new = []

    # --- Elitizm ---
    elite = best[0]
    for _ in range(5):
        new.append(elite.copy())

    # --- Parent havuzu: %10 ---
    top_pool = best[: max(2, population_size // 10)]

    while len(new) < population_size:
        p1 = np.random.choice(top_pool)
        p2 = np.random.choice(top_pool)
        
        child = NeuralNetwork.crossover(p1, p2)
        child.mutate()
        new.append(child)

    return new

# ----------------------------------------------------------
# ANA EĞİTİM DÖNGÜSÜ
# ----------------------------------------------------------

def main():
    population = [NeuralNetwork(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE) for _ in range(POPULATION_SIZE)]
    best_scores = []
    avg_scores = []
    gens = []

    for gen in range(1, MAX_GENERATIONS + 1):
        fitnesses = []
        scores = []

        for nn in population:
            fit, sc = evaluate_agent(nn)
            fitnesses.append(fit)
            scores.append(sc)

        best = max(scores)
        avg = np.mean(scores)

        best_scores.append(best)
        avg_scores.append(avg)
        gens.append(gen)

        print(f"Gen {gen:3d} | Best Score: {best:3d} | Avg: {avg:.2f}")

        best_agents = select_best(population, fitnesses)
        if gen < MAX_GENERATIONS:
            population = next_generation(best_agents, POPULATION_SIZE)

    # Grafik kaydet
    plt.figure(figsize=(10,6))
    plt.plot(gens, best_scores, label="Best Score")
    plt.plot(gens, avg_scores, label="Average Score")
    plt.legend()
    plt.savefig("ga_training.png")

    print("Training Complete!")

    # En iyi 5'i izleme
    top5 = best_agents[:5]
    while True:
        cmd = input("\nWatch the top agent? (y/n): ").lower()
        if cmd != "y":
            break
        for i, agent in enumerate(top5):
            print(f"> Agent #{i+1}")
            _, s = evaluate_agent(agent, render=True)
            print("Score:", s)

if __name__ == "__main__":
    main()
