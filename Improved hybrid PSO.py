import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import random

sns.set_style("whitegrid")
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

# seed of random number
np.random.seed(1234)

# parameters
city_num = 80  # number of cities
size = 100  # size of population
r1_s = 0.7  # pbest-xi 's initial retention probability
r2_s = 0.55  # gbest-xi 's initial retention probability
rm = 0.9  # probability of mutation
iter_num = 70  # max iteration number
fitneess_value_list = []  # each iteration 's best solution

# randomly generate city coordinates
X = np.random.choice(list(range(1, 100)), size=city_num, replace=False)
Y = np.random.choice(list(range(1, 100)), size=city_num, replace=False)


# calculate the distance between cities
def calculate_distance(X, Y):
    """
    calculate the Euclidean distance between two cities and store the result in a numpy matrix
    :param X: X-coordinate of the city, np.array
    :param Y: Y-coordinate of the city, np.array
    """
    distance_matrix = np.zeros((city_num, city_num))
    for i in range(city_num):
        for j in range(city_num):
            if i == j:
                continue
            dis = np.sqrt((X[i] - X[j]) ** 2 + (Y[i] - Y[j]) ** 2)
            distance_matrix[i][j] = dis
    return distance_matrix

def fitness_func(distance_matrix, xi):
    """
    fitness function，calculate fitness value.
    :param distance: distance matrix of cities
    :param xi: one solution of PSO
    :return: total distance
    """
    total_distance = 0
    for i in range(1, city_num):
        start = xi[i - 1]
        end = xi[i]
        total_distance += distance_matrix[start][end]
    total_distance += distance_matrix[end][xi[0]]  # go back to the starting city
    return total_distance

def plot_tsp(gbest):
    """draw the chart of the best solution"""
    plt.scatter(X, Y, color='r')
    for i in range(1, city_num):
        start_x, start_y = X[gbest[i - 1]], Y[gbest[i - 1]]
        end_x, end_y = X[gbest[i]], Y[gbest[i]]
        plt.plot([start_x, end_x], [start_y, end_y], color='b', alpha=0.8)
    start_x, start_y = X[gbest[0]], Y[gbest[0]]
    plt.plot([start_x, end_x], [start_y, end_y], color='b', alpha=0.8)

#  Exchange sequence function
def get_ss(xbest, xi, r):  # r=r1 or r2
    """
    calculate the exchange subsequence，corresponds to the PSO speed update equation 's part :
    r1(pbest-xi) and r2(gbest-xi)
    :param xbest: pbest or gbest
    :param xi: current solution
    :return:
    xbest
    """
    velocity_ss = []
    for i in range(len(xi)):
        if xi[i] != xbest[i]:
            j = np.where(xi == xbest[i])[0][0]  # find the differences of access order 's index
            so = (i, j, r)  # get swap operator, representing operate on i and j at the probability of r
            velocity_ss.append(so)
            xi[i], xi[j] = xi[j], xi[i]  # perform exchange
    return velocity_ss  # return the exchange subsequence

def do_ss(xi, ss):
    """
    perform swap
    :param xi:
    :param ss:
    :return:
    """
    for i, j, r in ss:
        rand = np.random.random()
        if rand <= r:
            xi[i], xi[j] = xi[j], xi[i]
    return xi

def mutate(xi,rm):
    rist = range(1,len(xi)-2)
    chosen = random.sample(rist,2)
    if np.random.random() <= rm:
        xi[chosen[0]], xi[chosen[1]] = xi[chosen[1]], xi[chosen[0]]
    return xi

# calculate distance matrix
distance_matrix = calculate_distance(X, Y)

XX = np.zeros((size, city_num), dtype=np.int)
for i in range(size):
    XX[i] = np.random.choice(list(range(city_num)), size=city_num, replace=False)

pbest = XX #initialize pbest
pbest_fitness = np.zeros((size, 1))
for i in range(size):
    pbest_fitness[i] = fitness_func(distance_matrix, xi=XX[i])

gbest = XX[pbest_fitness.argmin()]
gbest_fitness = pbest_fitness.min()


fitneess_value_list.append(gbest_fitness)

# cycles
for i in range(iter_num):  # iteration
    for j in range(size):  # particles
        pbesti = pbest[j].copy()
        xi = XX[j].copy()
        # calculate exchange orders
        r1 = r1_s+(0.75-r1_s)*np.exp((-i/iter_num))
        r2 = r2_s+(0.8-r2_s)*np.exp((-i/iter_num))
        ss1 = get_ss(pbesti, xi, r1)
        ss2 = get_ss(gbest, xi, r2)
        ss = ss1 + ss2
        # perform exchange
        temp_xi = xi
        xi = do_ss(xi, ss)
        # judge whether it is better
        fitness_new = fitness_func(distance_matrix, xi)
        fitness_old = pbest_fitness[j]
        if fitness_new <= fitness_old:
            pbest_fitness[j] = fitness_new
            pbest[j] = xi
        else:
            while np.random.random() < np.exp(-i / iter_num)+(1-(fitness_new/fitness_old)):
                xi = temp_xi

        # perform mutation
        temp_xim = xi
        xi = mutate(xi,rm)
        fitness_newm = fitness_func(distance_matrix, xi)
        fitness_oldm = pbest_fitness[j]
        if fitness_new < fitness_old:
            pbest_fitness[j] = fitness_new
            pbest[j] = xi
        else: xi = temp_xim

    # judge whether it is global optima and record iteration effects
    # fitneess_value_list.append(gbest_fitness)

    gbest_fitness_new = pbest_fitness.min()
    gbest_new = pbest[pbest_fitness.argmin()]
    if gbest_fitness_new < gbest_fitness:
        gbest_fitness = gbest_fitness_new
        gbest = gbest_new
    fitneess_value_list.append(gbest_fitness)

# print result
print('The optimal result of the iteration is：', gbest_fitness)
print('The optimal variables of the iteration are：', gbest)

# draw TSP 's path chart
plot_tsp(gbest)
plt.title('TSP path planning results')
plt.show()