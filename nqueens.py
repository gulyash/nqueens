import random


class Solver_8_queens:
    def __init__(self, pop_size=4, cross_prob=0.95, mut_prob=0.10):
        self.pop_size = pop_size
        self.cross_prob = cross_prob
        self.mut_prob = mut_prob
        self.desk_size = 8

    def solve(self, min_fitness=0.9, max_epochs=1000):
        self.epoch_num = 0
        self.min_fitness = min_fitness
        self.max_epochs = max_epochs
        self.population = self.generate_population()
        is_solved = False

        while not is_solved and self.epoch_num < self.max_epochs:
            self.parent_pool = self.roulette()
            self.crossover()
            self.mutate()
            self.epoch_num += 1
            is_solved, solution = self.check()
            self.population.sort(key=lambda b: b.fitness())
            solution = self.population[0]
            print('Epoch: ', self.epoch_num, '; ')
            for board in self.population:
                print(board.arrangement, end=' ')
            print('\nBest fitness: ', self.population[0].fitness())

        self.visualization = solution.arrangement
        self.best_fit = solution.fitness()
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
        self.children = children

    def child_chromosome(self, parent_a, parent_b):
        k = random.randrange(1, self.desk_size - 1)
        return parent_a.bits[:k] + parent_b.bits[k:]

    def roulette(self):
        parent_pool = []
        for _ in range(len(self.population)):
            pick = random.random()
            num = 0
            upper_bound = self.population[num].fitness() / sum(b.fitness() for b in self.population)
            while pick > upper_bound:
                pick -= upper_bound
                num += 1
                upper_bound = self.population[num].fitness() / sum(b.fitness() for b in self.population)
            parent_pool.append(self.population[num])
        return parent_pool

    def mutate(self):
        for child in self.children:
            prob = random.random()
            if prob < self.mut_prob:
                k = random.randrange(len(child.bits))
                if child.bits[k] == '0':
                    child.bits = child.bits[:k] + '1' + child.bits[k+1:]
                else:
                    child.bits = child.bits[:k] + '0' + child.bits[k+1:]
                child.arrangement = child.bits_to_arr()

    def check(self):
        for child in self.children:
            if child.fitness() < self.min_fitness:
                return True, child
        self.population = self.children
        self.children = []
        return False, None



class Board:
    def __init__(self, size=8, arrangement = None, bits = None):
        if arrangement is None and bits is None:
            self.arrangement = ''.join(str(random.randrange(size)) for _ in range(size))
            self.bits = self.arr_to_bits()
        elif not arrangement is None:
            self.arrangement = arrangement
            self.bits = self.arr_to_bits()
        elif not bits is None:
            self.bits = bits
            self.arrangement = self.bits_to_arr()

    def arr_to_bits(self):
        return ''.join(str(bin(int(num)))[2:].zfill(3) for num in self.arrangement)

    def bits_to_arr(self):
        list_of_cells_bin = [self.bits[i: i+3] for i in range (0, len(self.bits), 3)]
        return ''.join(map(lambda x: str(int(x, 2)), list_of_cells_bin))

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

        return conflicts