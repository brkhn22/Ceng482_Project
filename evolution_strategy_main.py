import numpy as np
import matplotlib.pyplot as plt
from game import SnakeGameAI, Direction, Point, BLOCK_SIZE
import time

# === OPTİMİZE EDİLMİŞ ES HİPERPARAMETRELERİ ===
MU = 50                     # Ebeveyn sayısı (stabil popülasyon)
LAMBDA = 350                # Offspring (mu*7 oranı korunuyor)
MAX_GENERATIONS = 200

INPUT_SIZE = 17
HIDDEN_SIZE = 32            
OUTPUT_SIZE = 3

# Mutasyon parametreleri - optimize edilmiş
INITIAL_SIGMA = 0.25        # Çok daha stabil başlangıç
MIN_SIGMA = 0.05            # Çok düşük olmamalı ama arama kaybolmasın
MAX_SIGMA = 0.8             # Aşırı kaos engellenir
TAU = 0.03                  # Self-adaptive sigma rate (optimal aralık)

class ESNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Başlangıç ağırlıkları
        self.weights1 = np.random.uniform(-1, 1, (input_size, hidden_size))
        self.bias1 = np.random.uniform(-1, 1, (1, hidden_size))
        self.weights2 = np.random.uniform(-1, 1, (hidden_size, output_size))
        self.bias2 = np.random.uniform(-1, 1, (1, output_size))
        
        # Self-adaptive mutation sigma
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
        # === Self-adaptive sigma update ===
        self.sigma *= np.exp(TAU * np.random.randn())
        self.sigma = max(MIN_SIGMA, min(self.sigma, MAX_SIGMA))

        # Mutasyon fonksiyonu
        def mutate(mat):
            return mat + np.random.randn(*mat.shape) * self.sigma

        # Apply mutation
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


# === GELİŞMİŞ STATE SİSTEMİ (16 input) ===
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

    state = []

    # === 1. Anlık Tehlike ===
    danger_straight = 0
    danger_right = 0
    danger_left = 0

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
    elif dir_d:
        danger_straight = game.is_collision(point_d)
        danger_right = game.is_collision(point_l)
        danger_left = game.is_collision(point_r)

    state.extend([int(danger_straight), int(danger_right), int(danger_left)])

    # === 2. Raycasting (3 input) ===
    if dir_r: directions = [Point(BLOCK_SIZE, 0), Point(0, BLOCK_SIZE), Point(0, -BLOCK_SIZE)]
    elif dir_l: directions = [Point(-BLOCK_SIZE, 0), Point(0, -BLOCK_SIZE), Point(0, BLOCK_SIZE)]
    elif dir_u: directions = [Point(0, -BLOCK_SIZE), Point(BLOCK_SIZE, 0), Point(-BLOCK_SIZE, 0)]
    elif dir_d: directions = [Point(0, BLOCK_SIZE), Point(-BLOCK_SIZE, 0), Point(BLOCK_SIZE, 0)]
    
    for direction in directions:
        dist = 0
        p = Point(head.x, head.y)
        for _ in range(20):
            p = Point(p.x + direction.x, p.y + direction.y)
            dist += 1
            if game.is_collision(p): break
        state.append(1.0 / dist)

    # === 3. Yön Bilgisi ===
    state.extend([int(dir_l), int(dir_r), int(dir_u), int(dir_d)])

    # === 4. Yemek Yönü ===
    food = game.food
    state.extend([
        int(food.x < head.x),
        int(food.x > head.x),
        int(food.y < head.y),
        int(food.y > head.y),
    ])

    # === 5. Space Awareness ===
    def measure_space(start, direction):
        space = 0
        p = start
        for _ in range(5):
            p = Point(p.x + direction.x, p.y + direction.y)
            if game.is_collision(p): break
            space += 1
        return space / 5.0

    if dir_r: vec_left, vec_right = Point(0, -BLOCK_SIZE), Point(0, BLOCK_SIZE)
    elif dir_l: vec_left, vec_right = Point(0, BLOCK_SIZE), Point(0, -BLOCK_SIZE)
    elif dir_u: vec_left, vec_right = Point(-BLOCK_SIZE, 0), Point(BLOCK_SIZE, 0)
    else:      vec_left, vec_right = Point(BLOCK_SIZE, 0), Point(-BLOCK_SIZE, 0)

    state.append(measure_space(head, vec_left))
    state.append(measure_space(head, vec_right))

    return np.array(state, dtype=float)


# === AGENT EVALUATION ===
def evaluate_agent(nn, render=False, display_speed_multiplier=1):
    game = SnakeGameAI(w=640, h=480, render_mode=render, display_speed_multiplier=display_speed_multiplier)
    game.reset()

    steps = 0
    max_steps = 100

    while True:
        state = get_state(game)
        action = nn.predict(state)

        head = game.snake[0]
        food = game.food
        before = abs(head.x - food.x) + abs(head.y - food.y)

        reward, game_over, score = game.play_step(action)

        head = game.snake[0]
        food = game.food
        after = abs(head.x - food.x) + abs(head.y - food.y)

        steps += 1
        if score > 0:
            max_steps = 100 + score * 50

        if steps > max_steps:
            game_over = True

        if game_over:
            break

    fitness = score * 1000 + steps * 0.1
    if score == 0:
        fitness += max(0, 500 - after)

    return max(0.1, fitness), score


# === TRAINING LOOP (Optimize edilmiş) ===
def main():
    print("\n=== STARTING OPTIMIZED EVOLUTIONARY STRATEGIES ===")

    parents = [ESNeuralNetwork(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE) for _ in range(MU)]

    best_scores = []
    avg_scores = []
    gen_idx = []

    all_time_best = None
    all_time_best_score = -1

    stagnation = 0
    last_best = 0

    for gen in range(MAX_GENERATIONS):

        offspring = []
        for _ in range(LAMBDA):
            p = np.random.choice(parents)
            c = p.copy()
            c.es_mutate()
            offspring.append(c)

        fitnesses = []
        scores = []

        for agent in offspring:
            fit, score = evaluate_agent(agent)
            fitnesses.append(fit)
            scores.append(score)

        idx = np.argsort(fitnesses)[::-1]
        parents = [offspring[i] for i in idx[:MU]]

        best_score = max(scores)
        avg_score = np.mean(scores)

        if best_score > all_time_best_score:
            all_time_best_score = best_score
            all_time_best = offspring[np.argmax(scores)].copy()
            stagnation = 0
            print(f"\n>> NEW RECORD SCORE: {all_time_best_score}\n")

        if best_score <= last_best:
            stagnation += 1
        else:
            stagnation = 0

        last_best = best_score

        if stagnation > 15:
            print("!! STAGNATION DETECTED → Smooth Sigma Boost")
            for p in parents:
                p.sigma *= 1.2
                p.sigma = min(p.sigma, 0.6)
            stagnation = 0

        best_scores.append(best_score)
        avg_scores.append(avg_score)
        gen_idx.append(gen + 1)

        print(f"Gen {gen+1:3d} | Best: {best_score:3d} | Avg: {avg_score:6.2f} | Sigma: {parents[0].sigma:.3f}")

        if gen % 10 == 0:
            plt.figure(figsize=(10, 5))
            plt.plot(gen_idx, best_scores, label="Best")
            plt.plot(gen_idx, avg_scores, label="Avg", linestyle="--")
            plt.title("Optimized ES Training Progress")
            plt.xlabel("Generation")
            plt.ylabel("Score")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.savefig("training_progress.png")
            plt.close()

    print("\n=== TRAINING COMPLETE ===")
    print(f"All-time best score: {all_time_best_score}")

    while True:
        print("\n1. Watch All-Time Best")
        print("2. Watch Best 5 of Final Gen")
        print("n. Exit")

        cmd = input("Choose: ").strip().lower()
        if cmd == "n":
            break
        elif cmd == "1":
            fit, score = evaluate_agent(all_time_best, render=True, display_speed_multiplier=3)
            print(f"Final Score: {score}")
        elif cmd == "2":
            print("\nShowing top 5 agents from final generation...")
            num_to_show = min(5, len(parents))
            for i in range(num_to_show):
                print(f"\n--- Agent {i+1} (Rank {i+1}) ---")
                fit, score = evaluate_agent(parents[i], render=True, display_speed_multiplier=3)
                print(f"Final Score: {score}")


if __name__ == "__main__":
    main()

