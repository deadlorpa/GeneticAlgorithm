import math
import random

import numpy
import numpy as np

'''
    Вариант 12 - x^2 *(sin(x-2,75)*cos(x+5)+ sin(3*x))
    Критерий: 3 - Достижение некоторого априорно заданного значения целевой функции.
                  На самом деле, я тут ещё и количество поколений ограничил
    Селекция: 2 - Схема пропорционального отбора
    Скрещивание: 4 - Однородное
    Мутация: 3 - Реверс битовой строки, начиная со случайно выбранного бита
'''
Q = 50  # количество поколений
MAX_ENT = 30  # размер популяции
N = 10  # длина хромосомы
LEFT_BORDER = 0  # левая граница
RIGHT_BORDER = 50  # правая граница
STEP_INTERVAL = math.fabs(RIGHT_BORDER - LEFT_BORDER) / (2**N - 1)  # шаг
MUTATION_PROBABILITY = 0.8  # вероятность мутации

EPSILON = 0.1
WANTED = 3937.78 # я видел такой экстремум на этом участке

fit_q = [[0 for _ in range(MAX_ENT)] for _ in range(Q)]  # лучший из поколения


def func(x):
    return math.fabs(x**2 *(math.sin(x-2.75)*math.cos(x+5)+ math.sin(3*x)))


def to_gray_code(n):
    n ^= (n >> 1)
    return bin(n)[2:].zfill(N)


def from_gray_code(n):
    gray_num = int(n, 2)
    bin = 0
    while gray_num > 0:
        bin ^= gray_num
        gray_num >>= 1
    return bin


def fitness(inp):  # inp – входной список из нулей и единиц
    x = to_phenotype(inp)
    # если вышли за пределы интервала
    if x > RIGHT_BORDER:
        x = RIGHT_BORDER
    return func(x) # ** (1 / 3)

def mutation(entity): # реверс сегмента по рандомному биту
    if random.random() <= MUTATION_PROBABILITY:
        random_bit = random.randint(0,len(entity)-1)
        tail = entity[random_bit:]
        entity = entity[:random_bit] + tail[::-1]
    return entity

def to_phenotype(gray_code):
    interval_ind = from_gray_code(gray_code)
    return (STEP_INTERVAL * interval_ind + STEP_INTERVAL * (interval_ind + 1)) / 2

def main():
    # Начальное поколение
    entity = [to_gray_code(i) for i in range(MAX_ENT)]
    q = 0
    while True:  # цикл поколений
        print(f"\nПоколение {q}")
        # Считаем fitness для каждого организма
        fit = [fitness(entity[i]) for i in range(MAX_ENT)]
        print('fit = ', max(fit))
        # Запомнили лучшего в поколении
        fit_q[q][:] = sorted(fit[:], reverse=True)

        if math.fabs(max(fit) - WANTED) <= EPSILON or q==Q-1:
            break

        fitness_avg = numpy.average(fit)
        # Родители, участвующие в скрещивании
        # пропорциональная история
        parents = []
        for i in range(MAX_ENT):
            chance = fit[i] / fitness_avg
            while chance > 1:
                parents.append(i)
                chance -= 1
            if random.random() <= chance:
                parents.append(i)

        # Воспроизводство
        child1 = ['0' for _ in range(N)]
        child2 = ['0' for _ in range(N)]
        k = 0
        # я не понимаю, в какой момент генерится маска, поэтому впихнул её сюда

        mask_breed_child1 = [random.randint(0,1) for i in range(N)]
        mask_breed_child2 = [random.randint(0, 1) for i in range(N)]
        while len(parents) > 1:
            f = True
            while f:
                m1, m2 = random.choices(parents, k=2)  # Родители
                if m1 == m2:
                    continue
                parents.remove(m1)
                parents.remove(m2)
                f = False
            # построение 2 потомков от m1 и m2
            for i in range(N):
                # 0 - первый родитель
                # 1 - второй родитель
                if mask_breed_child1[i] == 0:
                    child1[i] = entity[m1][i]
                else:
                    child1[i] = entity[m2][i]
                if mask_breed_child2[i] == 0:
                    child2[i] = entity[m1][i]
                else:
                    child2[i] = entity[m2][i]

            print('Родители {}  {}'.format(entity[m1], entity[m2]))
            print('Потомки {}  {} '.format("".join(child1), "".join(child2)))
            #print('Маски {} {} '.format(mask_breed_child1, mask_breed_child2))

            # добавка к массиву организмов двух потомков
            entity.append("".join(child1))
            entity.append("".join(child2))
            k += 1

        #  конец цикла воспроизводства

        # МУТАЦИИ
        for j in range(len(entity)):  # цикл по всем организмам вместе с потомками
            entity[j] = mutation(entity[j])
        # конец мутаций

        # ЕСТЕСТВЕННЫЙ ОТБОР
        chances = list(map(fitness, entity))
        chances_sum = sum(chances)
        chances = list(map(lambda ch: ch / chances_sum, chances))
        entity = list(np.random.choice(entity, size=MAX_ENT, p=chances))
        # конец цикла естественного отбора
        q += 1
    # конец цикла количества поколнеий Q
    fit = [(to_phenotype(entity[i]), fitness(entity[i])) for i in range(MAX_ENT)]
    best = max(fit, key=lambda fit_tuple: fit_tuple[1])
    for i in fit_q:
        print(i)
    print(f"Результат: {best[0]}. Значение функции: {func(best[0])}")
    print(f"Решение получено к {q+1} поколению [0...49]")

main()
