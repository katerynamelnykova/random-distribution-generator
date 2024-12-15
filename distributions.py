import math
import matplotlib.pyplot as plt
import random


class DistRand():
    def __init__(self, seed: int = None) -> None:
        self.seed = int(seed) if seed else random.randint(10**4, 10**5)
        self.m = 2**32
        self.a = 1664525
        self.c = 1013904223


    def normal_random(self, mu: float, sigma: float, n: int) -> list:
        random_values = []
        iterations_num = n // 2 if n % 2 == 0 else n // 2 + 1
        seed = self.seed
        for i in range(iterations_num): 
            seed = (self.a * seed + self.c) % self.m
            phi = seed / self.m
            seed = (self.a * seed + self.c) % self.m
            r = seed / self.m
            Z1 = math.sqrt(-2 * math.log(phi)) * math.cos(2 * math.pi * r)
            Z2 = math.sqrt(-2 * math.log(phi)) * math.sin(2 * math.pi * r)
            random_values.append(mu + sigma * Z1)

            if (n % 2 != 0 and i != iterations_num - 1) or n % 2 == 0:
                random_values.append(mu + sigma * Z2)
        return random_values[:n]
    
    def uniform_distribution(self, a: float, b: float, n: int) -> list:
        random_values = []
        seed = self.seed
        for _ in range(n):
            seed = (self.a * seed + self.c) % (self.m)
            r = seed / self.m
            X = a + r * (b - a)
            random_values.append(X)
        return random_values
    
    def exponential_distribution(self, la: float, n: int) -> list:
        random_values = []
        seed = self.seed
        for _ in range(n):
            seed = (self.a * seed + self.c) % (self.m)
            r = seed / self.m
            X = - math.log(r) / la
            random_values.append(X)
        return random_values
    
    def plot_data(self, data: list, title: str = ""):
        plt.figure(figsize=(10, 6))
        plt.hist(data, bins=30, density=True, alpha=0.7, color='blue', edgecolor='black')
        plt.title(title)
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.grid(True)
        plt.show()