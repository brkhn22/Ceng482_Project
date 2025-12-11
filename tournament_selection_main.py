import numpy as np
import matplotlib.pyplot as plt
from game import SnakeGameAI, Direction, Point, BLOCK_SIZE
from brain import NeuralNetwork

# --- OPTİMİZE EDİLMİŞ AYARLAR ---
POPULATION_SIZE = 500      
MUTATION_RATE = 0.05       
MAX_GENERATIONS = 100      
# INPUT_SIZE ARTIRILDI: 11 (Eski) + 3 (Mesafe) = 14 Girdi
INPUT_SIZE = 14            
HIDDEN_SIZE = 24           
OUTPUT_SIZE = 3
TOURNAMENT_SIZE = 3        


def get_state(game):
    """
    HİBRİT GÖRÜŞ SİSTEMİ:
    Hem eski 'Tehlike Var/Yok' (0/1) bilgisini,
    Hem de yeni 'Duvara Mesafe' (0.0-1.0) bilgisini birleştirir.
    Böylece yılan hem ani refleks gösterir hem de uzağı görür.
    """
    head = game.snake[0]
    
    # Yardımcı noktalar (Eski sistem)
    point_l = Point(head.x - BLOCK_SIZE, head.y)
    point_r = Point(head.x + BLOCK_SIZE, head.y)
    point_u = Point(head.x, head.y - BLOCK_SIZE)
    point_d = Point(head.x, head.y + BLOCK_SIZE)

    dir_l = game.direction == Direction.LEFT
    dir_r = game.direction == Direction.RIGHT
    dir_u = game.direction == Direction.UP
    dir_d = game.direction == Direction.DOWN

    state = []

    # --- 1. KISIM: ANLIK TEHLİKE (BOOLEAN - ESKİ SİSTEM) ---
    # Bu kısım yılanın hayatta kalmasını garantiye alır (Skor 40'a taşıyan kısım)
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

    # --- 2. KISIM: UZAK MESAFE GÖRÜŞÜ (RAYCASTING - YENİ SİSTEM) ---
    # Bu kısım yılanın tuzaklara girmesini engeller (Skor 100 için gerekli)
    
    # Bakış yönleri: [Düz, Sağ, Sol]
    if dir_r:
        directions = [Point(BLOCK_SIZE, 0), Point(0, BLOCK_SIZE), Point(0, -BLOCK_SIZE)]
    elif dir_l:
        directions = [Point(-BLOCK_SIZE, 0), Point(0, -BLOCK_SIZE), Point(0, BLOCK_SIZE)]
    elif dir_u:
        directions = [Point(0, -BLOCK_SIZE), Point(BLOCK_SIZE, 0), Point(-BLOCK_SIZE, 0)]
    elif dir_d:
        directions = [Point(0, BLOCK_SIZE), Point(-BLOCK_SIZE, 0), Point(BLOCK_SIZE, 0)]
    
    for direction in directions:
        distance = 0
        current_point = Point(head.x, head.y)
        # En fazla 20 blok uzağa bak (Sonsuz döngüyü ve aşırı işlemi önlemek için)
        for _ in range(20):
            current_point = Point(current_point.x + direction.x, current_point.y + direction.y)
            distance += 1
            if game.is_collision(current_point):
                break
        
        # Mesafeyi normalize et. Eğer çok uzaksa (20+), değer 0'a yakın olur.
        # Dibindeyse (1), değer 1.0 olur.
        state.append(1.0 / distance)

    # --- 3. KISIM: YÖN VE YEMEK (STANDART) ---
    state.extend([int(dir_l), int(dir_r), int(dir_u), int(dir_d)])
    
    state.append(int(game.food.x < head.x)) # Food Left
    state.append(int(game.food.x > head.x)) # Food Right
    state.append(int(game.food.y < head.y)) # Food Up
    state.append(int(game.food.y > head.y)) # Food Down
    
    return np.array(state, dtype=float)


def evaluate_agent(neural_net, render=False, display_speed_multiplier=1):
    game = SnakeGameAI(w=640, h=480, render_mode=render, display_speed_multiplier=display_speed_multiplier)
    game.reset()
    
    steps = 0
    max_steps = 100 
    
    while True:
        state = get_state(game)
        action = neural_net.predict(state)
        
        # Mesafe takibi (Eski ve güvenilir fitness için)
        head = game.snake[0]
        food = game.food
        dist_before = abs(head.x - food.x) + abs(head.y - food.y)
        
        reward, game_over, score = game.play_step(action)
        
        head = game.snake[0]
        food = game.food
        dist_after = abs(head.x - food.x) + abs(head.y - food.y)
        
        steps += 1
        
        if score > 0:
            max_steps = 100 + (score * 50)
            
        if steps > max_steps:
            game_over = True
            
        if game_over:
            break
            
    # --- FITNESS FONKSİYONU (Eski haline döndürüldü - DAHA KARARLI) ---
    # 1. Skor (Ana hedef)
    fitness = score * 1000
    
    # 2. Hayatta kalma (Küçük ödül)
    fitness += steps * 0.1
    
    # 3. Yemeğe Yaklaşma (Sadece hiç yememişse önemli)
    # Skor 0 ise bile yemeğe yaklaşan yaşasın.
    if score == 0:
        # Maksimum mesafe yaklaşık 1200 px. 
        # Yemeğe ne kadar yakın ölürse o kadar puan (Max 500).
        fitness += max(0, 500 - dist_after)

    return max(0.1, fitness), score


def tournament_selection(population, fitnesses, k=3):
    indices = np.random.choice(len(population), k, replace=False)
    best_idx = indices[0]
    best_fit = fitnesses[best_idx]
    for idx in indices[1:]:
        if fitnesses[idx] > best_fit:
            best_fit = fitnesses[idx]
            best_idx = idx
    return population[best_idx]


def create_next_generation(population, fitnesses, population_size, mutation_rate):
    new_population = []
    
    # Elitizm (Sıralama)
    sorted_indices = np.argsort(fitnesses)[::-1]
    
    # En iyi %5'i koru
    num_elites = max(1, int(population_size * 0.05))
    
    # Şampiyonu 3 kopya ekle (Garanti)
    champion = population[sorted_indices[0]]
    for _ in range(3): 
        new_population.append(champion.copy())
        
    for i in range(1, num_elites):
        idx = sorted_indices[i]
        new_population.append(population[idx].copy())

    # Turnuva Seçimi ve Çaprazlama
    while len(new_population) < population_size:
        parent1 = tournament_selection(population, fitnesses, k=TOURNAMENT_SIZE)
        parent2 = tournament_selection(population, fitnesses, k=TOURNAMENT_SIZE)
        
        child = NeuralNetwork.crossover(parent1, parent2)
        child.mutate(mutation_rate=mutation_rate, mutation_strength=0.2)
        
        new_population.append(child)

    return new_population


def create_population(size):
    return [NeuralNetwork(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE) for _ in range(size)]


def main():
    print(f"Starting HYBRID Genetic Algorithm...")
    print(f"Population: {POPULATION_SIZE}, Input Size: {INPUT_SIZE}, Hidden: {HIDDEN_SIZE}")
    
    population = create_population(POPULATION_SIZE)
    
    best_scores = []
    avg_scores = []
    generation_numbers = []
    
    for generation in range(MAX_GENERATIONS):
        fitnesses = []
        scores = []
        
        for nn in population:
            fitness, score = evaluate_agent(nn, render=False)
            fitnesses.append(fitness)
            scores.append(score)
        
        best_score = max(scores)
        avg_score = np.mean(scores)
        best_fitness = max(fitnesses)
        
        best_scores.append(best_score)
        avg_scores.append(avg_score)
        generation_numbers.append(generation + 1)
        
        print(f"Gen {generation+1:3d} | Best Score: {best_score:3d} | Avg: {avg_score:.2f} | Max Fit: {best_fitness:.1f}")
        
        if generation < MAX_GENERATIONS - 1:
            population = create_next_generation(population, fitnesses, POPULATION_SIZE, MUTATION_RATE)
            
        if generation % 10 == 0:
             plt.figure(figsize=(10, 5))
             plt.plot(generation_numbers, best_scores, label='Best Score')
             plt.plot(generation_numbers, avg_scores, label='Average Score')
             plt.legend()
             plt.grid(True, alpha=0.3)
             plt.savefig('training_progress.png')
             plt.close()

    print("Training Complete!")
    print(f"All Time Best Score: {max(best_scores)}")
    
    sorted_indices = np.argsort(fitnesses)[::-1]
    top_5_agents = [population[i] for i in sorted_indices[:5]]
    
    while True:
        cmd = input("\nWatch top 5 agents play? (y/n): ").strip().lower()
        if cmd == 'n': break
        for i, agent in enumerate(top_5_agents):
            print(f"\n>> Watching Agent Rank #{i+1}...")
            _, run_score = evaluate_agent(agent, render=True, display_speed_multiplier=1)
            print(f"   Agent Rank #{i+1} Finished! -> Final Score: {run_score}")

if __name__ == "__main__":
    main()