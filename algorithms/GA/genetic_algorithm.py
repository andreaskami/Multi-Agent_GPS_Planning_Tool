from utility.misc import get_path_distances, distance, get_polar_angle, polar_to_cartesian
from utility.file_io import get_project_root

from algorithms.two_opt import two_opt_path
from utility.vrp_utils import VRPDescriptor, cost

import random
import os
import matplotlib.pyplot as plt
import numpy as np

from time import time

runtime_default = 0


def genetic_algorithm(env, scenario, population_size=None, max_runtime=runtime_default, plot_history=False,
                      stop_thres=100000, phi=1., speed_scale=1.0, **kwargs):
    """
    A genetic algorithm for the vehicle routing problem - Baker & Ayechew, 2003
    Note: does not work for a single agent

    :param env: an instance of the MultiAgentEnv class, holds the World object.
    :param scenario: an instance of the Scenario class (see scenario.py).
    :param population_size: size of population. Recommended 30 for n < 50, and 50 for larger problems.
    :param max_runtime: Maximum time to keep running (seconds)
    :param plot_history: whether or not to output a plot of the fitness history.
    :param stop_thres: the number of iterations after which to stop if no improvement has been found
    :param phi: scaling factor to generate customer demands (see vrp_utils.py).
    :param speed_scale: scaling factor for max_distance (see vrp_utils.py).
    """
    vrp = VRPDescriptor(scenario, phi=phi, speed_scale=speed_scale)

    if vrp.n_vehicles <= 1:
        raise ValueError('GA expects at least two vehicles, otherwise its initial solutions will break.')

    if max_runtime == runtime_default:  # calculate time limit
        if vrp.n_customers <= 100:
            max_runtime = int(100**3 / 2**10)
        else:
            max_runtime = int(vrp.n_customers**3 / 2**10)

    if population_size is None:
        if vrp.n_customers > 50:
            population_size = 50  # 50 for larger problems
        else:
            population_size = 30  # 30 for smaller problems

    ga_solver = GA(env, scenario, vrp, population_size)
    paths = ga_solver.get_ga_solution(max_runtime, plot_history, stop_thres)

    return {'solution': paths, 'info': {'population_size': population_size, 'stop_thres': stop_thres}}


class GA:
    def __init__(self, env, scenario, vrp, population_size):
        self.population_size = population_size
        self.vrp = vrp
        self.tightness = sum(self.vrp.demand) / (self.vrp.n_vehicles * self.vrp.capacity)  # for acceptance decision
        self.rd_acceptance = 0.9 if self.tightness < 0.8 else 0.75

        # sort and number customers - they get a number for each vehicle [1, 2, 2, 3, 3, 4, 1, 1]
        #   for random init: sort by polar angle to the depot
        #   for cluster init: sort by NN solution through all points
        # number vehicles approximately in the same region

        self.split = int(round(self.population_size * 0.5))  # split point between sweep and generalized assignment init

        self.customers = [i for i in range(self.vrp.n_customers)]
        self.polar_angles = None

        self.locations = scenario.get_locations(env.world)
        self.depot_location = self.locations[self.vrp.depot]
        self.sort_customers_polar(self.locations)

        self.population = []
        self.fitness = []
        self.unfitness = []
        self.fitness_history = []
        self.unfitness_history = []

        self.best_fitness = float('inf')
        self.best_solution = []
        self.best_generation = 0  # last best generation - used for stopping criterion
        self.generation = 0  # current generation

    def get_ga_solution(self, max_runtime, plot_history, stop_thres):
        """Main GA loop."""
        self.get_initial_solution()
        start_time = time()
        time_limit = False
        while not time_limit:
            self.reproduce()
            if self.generation % self.population_size == 0:  # update history every pop_size number of new children
                self.fitness_history.append(min(self.fitness))  # best fitness
                self.unfitness_history.append(max(self.unfitness) / self.population_size)

            if self.generation - self.best_generation > stop_thres:
                break  # solution limit stopping criterion reached

            if time() - start_time > max_runtime:
                time_limit = True  # time limit exceeded - stopping criterion number 2

        if plot_history:
            self.plot_history()

        return self.vrp.match_route_lengths_and_wrap(self.transform_solution(self.best_solution, force_two_opt=True))

    def get_initial_solution(self):
        """Fill initial population."""
        # TODO all 0s in sweep when phi=0.? Maybe find an alternative for this
        if self.vrp.n_customers >= self.population_size:
            start_customers = random.sample(range(self.vrp.n_customers), self.population_size)
        else:
            start_customers = random.sample(range(self.population_size), self.population_size)
            start_customers = [c % self.vrp.n_customers for c in start_customers]

        for member in range(self.split):
            self.population.append(self.init_sweep_solution(start_customers[member]))

        for member in range(self.split, self.population_size):
            self.population.append(self.init_generalized_assignment_solution(start_customers[member]))

        for member in self.population:
            member_fitness, member_unfitness = self.calc_fitness(member)
            self.fitness.append(member_fitness)
            self.unfitness.append(member_unfitness)

    def reproduce(self):
        """
        Reproduction:
        perform 2-point crossover between parents (figure 2):
          choose 2 random points in each solution; swap between these points with other parent
          watch angles of vehicles! Perhaps renumber accordingly
          the other way around as well to produce second offspring. Discard existing pop. members

        Sswap 2 random customers as well, except if they already are on the same vehicle
        Apply uniform crossover if not all vehicles are used
        """
        self.generation += 1  # next generation, even if not accepted
        # select parents through binary tournament
        first_parent = self.tournament_selection()
        second_parent = self.tournament_selection()
        while first_parent == second_parent:  # make sure second parent is not again the first parent
            second_parent = self.tournament_selection()
        children = self.two_point_crossover([first_parent, second_parent])  # get two-point crossover children
        if set(first_parent + second_parent) != set(children[0] + children[1]):  # if both offspring do not use all v,
            children = [self.uniform_crossover([first_parent, second_parent])]  # replace offspring by single uniform c
        for offspring in children:
            self.simple_mutation(offspring)  # apply mutation to children
            if self.unique_offspring(offspring):  # check if child not already in population
                self.update_population(offspring)  # update population if valid - else, discard offspring

    def init_sweep_solution(self, start_customer):
        solution = [-1 for _ in range(len(self.customers))]
        idx = start_customer
        for v in range(self.vrp.n_vehicles):
            if idx == start_customer and v > 0:
                break
            route = [self.customers[idx]]
            solution[idx] = v
            valid, dist = self.vrp.valid_route(route)
            prev_dist = dist
            while valid:
                idx = (idx + 1) % len(self.customers)
                if idx == start_customer:
                    break
                route.append(self.customers[idx])
                solution[idx] = v  # will be overwritten later if invalid
                prev_dist = dist
                valid, dist = self.vrp.valid_route(route)
            rd = prev_dist / dist  # check "badness" of invalid route - close to one is "not that bad"
            if rd >= self.rd_acceptance and idx != start_customer:  # move to the next point
                idx = (idx + 1) % len(self.customers)
        # TODO this is a quick fix to avoid issues - solve later. Makes below assert statement obsolete.
        for i, s in enumerate(solution):
            if s == -1:
                solution[i] = 0
        assert sum([1 if s == -1 else 0 for s in solution]) == 0, 'Unassigned points in sweep initial solution!'
        return solution

    def init_generalized_assignment_solution(self, start_customer):
        """
        1. Create seeds - draw customer cones
        2. Draw vehicle cones for equal demand
        3. seed_v_dist d0i = max dist to depot from customers in cone
        """

        solution = [-1 for _ in range(len(self.customers))]

        seeds = self.generate_vehicle_seeds(start_customer)
        seed_dists = [distance(seed, self.depot_location) for seed in seeds]
        for c, customer in enumerate(self.customers):
            seed_savings = [self.vrp.distances[self.vrp.depot][customer] + distance(seed, self.locations[customer]) -
                            seed_dist for seed, seed_dist in zip(seeds, seed_dists)]

            if self.vrp.n_vehicles == 2:
                seed_idxs = seed_savings
            else:
                seed_idxs = np.argpartition(seed_savings, 2)[0:2]  # get best two candidate seeds
            best_seed, scnd_best_seed = [seed_savings[idx] for idx in seed_idxs]
            p = best_seed / (best_seed + scnd_best_seed)  # probability of which seed to choose
            choice = int(random.random() > p)
            solution[c] = seed_idxs[choice]  # if random > p, select second best seed. Else, best seed.
        assert sum([1 if s == -1 else 0 for s in solution]) == 0, 'Unassigned points in g.a. initial solution!'
        return solution

    def tournament_selection(self):
        """Select two random parents using a binary tournament."""
        targets = random.sample(range(self.population_size), 2)
        if self.fitness[targets[0]] < self.fitness[targets[1]]:
            return self.population[targets[0]]
        else:
            return self.population[targets[1]]

    def calc_fitness(self, solution):
        """Calculate solution fitness, using predefined cost function."""
        routes = self.transform_solution(solution)
        dists = get_path_distances(routes, self.vrp.distances)
        fitness = cost(dists)

        demand_excess, dist_excess = 0, 0
        for r, route in enumerate(routes):
            demand = self.vrp.route_demand(route)
            if demand > self.vrp.capacity:
                demand_excess += (demand - self.vrp.capacity) / self.vrp.capacity
            if dists['list'][r] > self.vrp.max_distance:
                dist_excess += (dists['list'][r] - self.vrp.max_distance) / self.vrp.max_distance

        unfitness = demand_excess + dist_excess
        return fitness, unfitness

    def transform_solution(self, solution, force_two_opt=False):
        """Transform GA solution into regular shape that is used with the other algorithms."""
        # TODO add 2-opt and 3-opt refinement
        paths = [[] for _ in range(self.vrp.n_vehicles)]
        for customer, vehicle in enumerate(solution):
            paths[vehicle].append(self.customers[customer])
        if self.vrp.n_customers <= 100 or force_two_opt:
            return [two_opt_path(route, self.vrp.depot, self.vrp.distances)[0] for route in paths]
        else:  # for larger problems, 2-opt is too computationally expensive
            return paths

    def update_population(self, offspring):
        """Check offspring fitness and unfitness, and if good enough, add it to population. See paper for more info."""
        subsets = [-1 for _ in range(3)]
        candidate_fitness = [0 for _ in range(3)]
        candidate_unfitness = [0 for _ in range(3)]

        child_fitness, child_unfitness = self.calc_fitness(offspring)
        for idx in range(self.population_size):
            if self.unfitness[idx] >= child_unfitness and self.fitness[idx] >= child_fitness:
                subset_idx = 0
            elif self.unfitness[idx] >= child_unfitness and self.fitness[idx] < child_fitness:
                subset_idx = 1
            elif self.unfitness[idx] < child_unfitness and self.fitness[idx] >= child_fitness:
                subset_idx = 2
            else:
                subset_idx = -1  # do nothing

            if subset_idx < 0:
                continue  # do not add child to population

            if self.unfitness[idx] > candidate_unfitness[subset_idx] or \
                    (self.unfitness[idx] == candidate_unfitness[subset_idx] and
                     self.fitness[idx] > candidate_fitness[subset_idx]):
                subsets[subset_idx] = idx  # this one is worse than all others in the subset

        for candidate in subsets:
            if candidate >= 0:  # replacement candidate found - overwrite population instance
                self.population[candidate] = offspring
                self.fitness[candidate] = child_fitness
                self.unfitness[candidate] = child_unfitness
                if child_fitness < self.best_fitness:  # store record solutions
                    self.best_solution = offspring
                    self.best_fitness = child_fitness
                    self.best_generation = self.generation
                return

    def plot_history(self):
        """Plot GA fitness history."""
        x = range(len(self.fitness_history))
        plt.plot(x, self.fitness_history, x, self.unfitness_history)
        plt.title('Average fitness and unfitness of population over time.')
        plt.xlabel('Generation')
        plt.legend(['Fitness', 'Unfitness'])
        plt.savefig(os.path.join(get_project_root(), 'im', 'ga_fitness_history.png'))

    @staticmethod
    def uniform_crossover(parents):
        """Perform uniform crossover: produce offspring by randomly grabbing a gene from each parent for every gene."""
        offspring = []
        for customer in range(len(parents[0])):
            offspring.append(parents[random.choice([0, 1])][customer])
        return offspring

    def two_point_crossover(self, parents):
        angles = self.avg_parent_angles(parents)
        if len(angles[0]) == 2 and len(angles[1]) == 2:
            if abs(angles[0][0] - angles[1][1]) < abs(angles[0][0] - angles[1][0]):
                parents[0] = self.renumber_parent_vehicles(parents[0], 1)
            if abs(angles[0][1] - angles[1][0]) < abs(angles[0][1] - angles[1][1]):
                parents[1] = self.renumber_parent_vehicles(parents[1], -1)

        left, right = sorted(random.sample(range(len(parents[0])), 2))
        assert left < right, 'Left swap index should be smaller than right swap index.'
        offspring_1 = parents[0][0:left] + parents[1][left:right] + parents[0][right:]
        offspring_2 = parents[1][0:left] + parents[0][left:right] + parents[1][right:]
        return offspring_1, offspring_2

    @staticmethod
    def simple_mutation(offspring):
        """Swaps two random customers provided they are not on the same route."""
        left, right = random.sample(range(len(offspring)), 2)
        assert left != right, 'Left swap index should not be equal to than right swap index.'
        if offspring[left] != offspring[right]:  # perform swap only if customers not on the same route
            offspring[left], offspring[right] = offspring[right], offspring[left]

    def unique_offspring(self, offspring):
        """Check if offspring does not exist in current population."""
        for parent in self.population:
            if offspring == parent:
                return False
        return True

    def sort_customers_polar(self, landmarks):
        """Gets polar angles, and sorts customers accordingly."""
        self.polar_angles = [get_polar_angle(self.depot_location, landmarks[c]) for c in self.customers]
        self.customers = [c for _, c in sorted(zip(self.polar_angles, self.customers))]
        self.polar_angles = np.asarray(sorted(self.polar_angles))

    def avg_parent_angles(self, parents):
        """Get average angles between parents."""
        avgs = [[], []]
        rep = 0
        for v in set(parents[0]) & set(parents[1]):
            if rep >= 2:
                break
            avgs[0].append(self.polar_angles[np.where(np.asarray(parents[0]) == v)].mean())
            avgs[1].append(self.polar_angles[np.where(np.asarray(parents[1]) == v)].mean())
            rep += 1
        return avgs

    def renumber_parent_vehicles(self, parent, shift):
        """Renumbers parent vehicles. shift should either be 1 or -1."""
        return [(c + shift) % self.vrp.n_vehicles for c in parent]

    def generate_vehicle_seeds(self, start_customer):
        """Generate equally distributed vehicle seeds for generalized assingment initial solution."""
        seeds = []
        # first cone is a special case
        first_cone = (self.polar_angles[0] + 2*np.pi + self.polar_angles[-1]) / 2
        first_cone = first_cone - 2*np.pi if first_cone >= np.pi else first_cone
        cones = [first_cone] + [(self.polar_angles[c] +
                                 self.polar_angles[(c - 1) % len(self.polar_angles)]) / 2
                                 for c in range(1, len(self.polar_angles))]

        if self.vrp.phi > 0:
            v_demand = sum(self.vrp.demand) / self.vrp.n_vehicles
        else:
            v_demand = self.vrp.n_customers / self.vrp.n_vehicles

        demand = 0
        idx = start_customer
        angle_left, angle_right = cones[idx], cones[idx]
        tol = 1e-12

        for v in range(self.vrp.n_vehicles):
            cone_customers = []
            while True:
                cone_customers.append(self.customers[idx])  # this customer is (at least partially) in this cone
                if self.vrp.phi > 0:
                    demand_inc = self.vrp.demand[self.customers[idx]]
                else:
                    demand_inc = 1  # replacement "demand" to assure equal distribution of cones
                demand_ = demand + demand_inc

                if demand_ > v_demand or abs(demand_ - v_demand) < tol:
                    diff = demand_ - v_demand  # find overshoot in demand

                    if abs(diff) < tol:  # customer fits exactly - catches float exception
                        angle_right = cones[(idx + 1) % len(cones)]
                        demand = 0  # set starting demand for next vehicle
                        idx = (idx + 1) % len(self.customers)  # move to next customer for next vehicle
                    else:
                        cone_left, cone_right = cones[idx], cones[(idx + 1) % len(cones)]
                        # catch wrap around with +/- pi
                        if cone_right < cone_left:
                            cone_right = cone_right + 2*np.pi
                        angle_right = cone_left + (cone_right - cone_left) * ((demand_inc-diff)/demand_inc)
                        demand = diff  # set starting demand for next vehicle

                    if angle_right < angle_left:  # wrap around for correct average
                        angle_right += 2*np.pi
                    angle = (angle_right + angle_left) / 2

                    customer_distances = [self.vrp.distances[self.vrp.depot][c] for c in cone_customers]
                    seeds.append(polar_to_cartesian(self.depot_location, angle, max(customer_distances)))

                    angle_left = angle_right if -np.pi <= angle_right < np.pi else angle_right - 2*np.pi
                    break

                else:  # continue loop
                    demand = demand_  # update demand
                    idx = (idx + 1) % len(self.customers)  # move to next customer cone

        return seeds





