import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # BAŞLANGIÇ AYARLARI (Uniform Random)
        # -1 ile 1 arasında rastgele değerler. Xavier kullanma!
        self.weights1 = np.random.uniform(-1, 1, (self.input_size, self.hidden_size))
        self.bias1 = np.random.uniform(-1, 1, (1, self.hidden_size))
        
        self.weights2 = np.random.uniform(-1, 1, (self.hidden_size, self.output_size))
        self.bias2 = np.random.uniform(-1, 1, (1, self.output_size))

    def relu(self, x):
        return np.maximum(0, x)
    
    def predict(self, input_data):
        # Inputu matrise çevir
        if len(input_data.shape) == 1:
            input_data = input_data.reshape(1, -1)
        
        # 1. Katman (Input -> Hidden)
        hidden = np.dot(input_data, self.weights1) + self.bias1
        hidden = self.relu(hidden)
        
        # 2. Katman (Hidden -> Output)
        output = np.dot(hidden, self.weights2) + self.bias2
        
        # En büyük değeri seç (Argmax)
        action_idx = np.argmax(output)
        
        # One-hot encoding döndür [1,0,0] gibi
        action = [0, 0, 0]
        action[action_idx] = 1
        return action
    
    def mutate(self, mutation_rate=0.05, mutation_strength=0.2):
        # Ağırlıkları belirli bir olasılıkla (rate) değiştir
        def mutate_matrix(matrix):
            mask = np.random.random(matrix.shape) < mutation_rate
            noise = np.random.randn(*matrix.shape) * mutation_strength
            matrix[mask] += noise[mask]

        mutate_matrix(self.weights1)
        mutate_matrix(self.bias1)
        mutate_matrix(self.weights2)
        mutate_matrix(self.bias2)
    
    def copy(self):
        new_nn = NeuralNetwork(self.input_size, self.hidden_size, self.output_size)
        new_nn.weights1 = np.copy(self.weights1)
        new_nn.bias1 = np.copy(self.bias1)
        new_nn.weights2 = np.copy(self.weights2)
        new_nn.bias2 = np.copy(self.bias2)
        return new_nn

    @staticmethod
    def crossover(parent1, parent2):
        child = NeuralNetwork(parent1.input_size, parent1.hidden_size, parent1.output_size)
        
        # %50 şansla genleri karıştır
        def mix_genes(w1, w2):
            mask = np.random.rand(*w1.shape) < 0.5
            return np.where(mask, w1, w2)

        child.weights1 = mix_genes(parent1.weights1, parent2.weights1)
        child.bias1 = mix_genes(parent1.bias1, parent2.bias1)
        child.weights2 = mix_genes(parent1.weights2, parent2.weights2)
        child.bias2 = mix_genes(parent1.bias2, parent2.bias2)
        
        return child
