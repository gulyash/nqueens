import random


class Solver_8_queens:
    def __init__(self, pop_size=250, cross_prob=0.95, mut_prob=0.10):
        self.pop_size = pop_size
        self.cross_prob = cross_prob
        self.mut_prob = mut_prob
        self.desk_size = 8
        self.bit_num = 24

    def solve(self, min_fitness=27.5, max_epochs=1000):
        self.epoch_num = 0
        self.min_fitness = min_fitness
        self.max_epochs = max_epochs
        self.population = self.generate_population()

        is_solved = False
        while not is_solved and self.epoch_num < self.max_epochs:
            # self.parent_pool = self.roulette()
            # self.new_generation = self.crossover()
            # self.new_generation = self.mutate(self.new_generation)
            self.epoch_num += 1
            # self.population = self.new_generation
            # self.new_generation = []
            is_solved, solution = self.check()
        if solution is None:
            solution = self.population[0]


        self.visualization = solution.visualization()
        self.best_fit = solution.fitness
        return self.best_fit, self.epoch_num, self.visualization

    def generate_population(self):
        population = []
        for _ in range(self.pop_size):
            population.append(Board(self.desk_size))
        return population

    def crossover(self):
        children = []
        while len(children) < len(self.population):
            pair = random.sample(self.parent_pool, 2)
            pick = random.random()
            if pick < self.cross_prob:
                child = Board(bits=self.child_chromosome(*pair))
                children.append(child)
        return children

    def child_chromosome(self, parent_a, parent_b):
        k = random.randrange(1, self.desk_size - 1)
        return parent_a.bits[:k] + parent_b.bits[k:]

    def roulette(self):
        parent_pool = []
        fitness_sum = sum(b.fitness for b in self.population)
        for _ in range(len(self.population)):
            pick = random.random()
            num = 0
            upper_bound = self.population[num].fitness / fitness_sum
            while pick > upper_bound:
                pick -= upper_bound
                num += 1
                upper_bound = self.population[num].fitness / fitness_sum
            parent_pool.append(self.population[num])
        return parent_pool

    def mutate(self, pop):

        for child in pop:
            prob = random.random()
            if prob < self.mut_prob:
                k = random.randrange(len(child.bits))
                if child.bits[k] == '0':
                    child.bits = child.bits[:k] + '1' + child.bits[k + 1:]
                else:
                    child.bits = child.bits[:k] + '0' + child.bits[k + 1:]
                child.arrangement = child.bits_to_arr()
        return pop

    def check(self):
        self.population.sort(key=lambda b: b.fitness, reverse=True)
        best_board = self.population[0]
        return best_board.fitness > self.min_fitness, best_board


class Board:
    def __init__(self, size=8, arrangement=None, bits=None):
        if arrangement is None and bits is None:
            self.arrangement = ''.join(str(random.randrange(size)) for _ in range(size))
            self.bits = self.arr_to_bits()
        elif arrangement is not None:
            self.arrangement = arrangement
            self.bits = self.arr_to_bits()
        elif bits is not None:
            self.bits = bits
            self.arrangement = self.bits_to_arr()
        self.fitness = self.fitness()

    def arr_to_bits(self):
        return ''.join(str(bin(int(num)))[2:].zfill(3) for num in self.arrangement)

    def bits_to_arr(self):
        list_of_cells_bin = [self.bits[i: i + 3] for i in range(0, len(self.bits), 3)]
        return ''.join(map(lambda x: str(int(x, 2)), list_of_cells_bin))

    def gray(self):
        gray = [int(self.bits[0], 2)]
        for i in range(1, len(self.bits)):
            gray.append(int(self.bits[i-1]) ^ int(self.bits[i]))
        return ''.join(map(str, gray))

    def fitness(self):
        conflicts = 0
        board = self.arrangement
        for i in range(len(board) - 1):

            for j in range(i + 1, len(board)):
                if board[j] == board[i]:
                    conflicts += 1
                right_lower_diag = str(int(board[i]) + j - i)
                right_upper_diag = str(int(board[i]) - j + i)
                if board[j] == right_lower_diag:
                    conflicts += 1
                if board[j] == right_upper_diag:
                    conflicts += 1

        return 28 - conflicts

    def visualization(self):
        s = ''
        for i in self.arrangement:
            s = s + '+' * int(i) + 'Q' + '+'*(len(self.arrangement) - 1 - int(i)) + '\n'
        return s
