"""
Analysis_simulation
Lior Sinai, 23 August 2019
Adapted for use with MultiAgentEnv by Jochem Postmes, November 2020

Train using the Ant Colony Optimisation algorithm, for a multiple ants.
Each ant has its own pheromone map, probability matrix and memory

Sources:
1) M.Dorigo, Ant Colony Optimization: Overview and Recent Advances, 2019
2) F. Zhao et al, An Ant Colony Optimization Algorithm for the Multiple Traveling Salesmen Problem, 2009

procedure AntColonyOptimisation:
while(not_termination)
   generateSolutions()
     - choose probabilistic transition based on a desirability weight (eta) and artificial pheromone (previous solution)
       weighting (tau)
     - prob_matrix = [tau**alpha]*[eta**beta]/sum([tau**alpha]*[eta**beta]) , eta = 1/distance(i,j)
   daemonActions()
     - local search is recommended
   globalPheromoneUpdate()
     - del_tau = Q/length_tour
     - tau = (1 - evap_coeff)* tau;  tau(i,j) += del_tau
end while
end procedure

Use max-min ant system scheme
- maximum and minimum pheromone levels, tau_max = 1/(evap_coeff*L_best), tau_min = tau_max/(2*n)
- update only the pheromone level of the best solution (best tour) but use the current tour length to
  provide some randomness to the update (reduces convergence to local minimum)

"""

from algorithms.ACO.ACO_utils import update_prob_matrix, global_pheromone_update, get_targets_prob, get_actions_sequences
import numpy as np


def train_ACO(env, ant_origins: np.ndarray, cities: np.ndarray, scenario, *,
              alpha: float, beta: float, evap_coeff: float, q: float,
              tol: float, n_mean=20, max_iter=100,
              share_pheromone=False, mode=0):
    """
    Ant Colony Optimisation for a single ant
    Inputs:
    :param env: an environment object
    :param ant_origins: nr_ants x 3 array with the starting locations in 2D (n x 2) and velocities (n x 1)
    :param cities: nr_cities x 2 array with the city locations in 2D
    :param scenario: MultiAgentEnv Scenario object
    :param alpha: ACO control parameter
    :param beta: ACO control parameter
    :param evap_coeff: ACO control parameter - evaporation coefficient
    :param q: ACO control parameter
    :param tol: convergence based on a running mean of the lengths compared to the optimal solution
    :param n_mean: number of latest samples considered in the running mean
    :param max_iter: maximum number of iterations
    :param mode: combined_cost = c_max * mode + (1 - mode)*c_total/
    :param share_pheromone: share pheromone between ants
    """
    assert 0 <= mode <= 1, "Mode is outside the range [0, 1]"

    # ------------------- Initialise ------------------- #
    nr_cities = scenario.num_landmarks + scenario.num_spawns  # add 1 for the nest
    nr_ants = env.n
    cities_ = np.vstack((cities, ant_origins))  # add the origins, so that the indices match with world.landmarks
    origin_idx = scenario.num_landmarks
    # Check if distance matrix is not empty
    if scenario.distances is None:
        raise ValueError('Scenario distance matrix is None object.')

    # desirability matrix from any city to any other city
    ind_triu = np.triu_indices(nr_cities, k=1)
    eta = np.zeros((nr_cities, nr_cities))
    for edge in zip(ind_triu[0], ind_triu[1]):
        eta[edge] = 1/scenario.distances[edge]
    eta_beta = np.power(eta[ind_triu], beta)
    # pheromone matrix
    if share_pheromone:
        nr_tau = 1
    else:
        nr_tau = env.n
    tau_lim = [1/(evap_coeff*2*nr_cities), 1/evap_coeff]  # [min, max]
    tau = np.zeros((nr_tau, nr_cities, nr_cities))
    for k in range(nr_tau):
        tau[k][ind_triu] = tau_lim[1]  # set upper triangle to max pheromone value
    # probability matrix
    if share_pheromone:
        prob_matrix = [np.zeros((nr_cities, nr_cities))]*env.n  # linked copies
    else:
        prob_matrix = np.zeros((env.n, nr_cities, nr_cities))
    ind_tril = np.tril_indices(nr_cities, k=-1)
    ind_invalid = scenario.invalid_edges
    # stats
    cost_opt, length_opt, slowest_agent = float('inf'), float('inf'), float('inf')
    costs = []
    n_opt = 0
    def combined_cost(c_total, c_max): return c_max * mode + (1 - mode) * c_total / nr_ants

    completed = False
    trails_opt = None
    for iters in range(0, max_iter):
        # ------------------- generate solutions  ------------------- #
        env.reset()
        targets = [[] for _ in range(nr_ants)]  # reset targets
        for k in range(nr_ants):
            scenario.targets[k] = [origin_idx]  # reset history
        for k in range(nr_tau):
            update_prob_matrix(prob_matrix[k], tau=tau[k], eta_beta=eta_beta, alpha=alpha,
                               ind_triu=ind_triu, ind_tril=ind_tril, ind_invalid=ind_invalid)

        t = -1
        while ~completed:
            t += 1
            visited_landmarks, visited_spawns = scenario.get_visited(env.world)
            visited_total = visited_landmarks + visited_spawns
            if sum(visited_landmarks) >= scenario.num_landmarks:
                break

            targets = get_targets_prob(env.world.agents, targets, visited=visited_landmarks, origin_idx=origin_idx,
                                       n_nests=scenario.num_spawns, prob_matrix=prob_matrix, rng=np.random,
                                       min_tour_length=1)

            # TODO remove call and definition of get_action_sequences. Deprecated, no longer used
            actions = get_actions_sequences(targets, agents=env.world.agents, points=cities_,
                                            visited=visited_total, dynamics='direct')

            scenario.set_targets(env.world, targets=targets)
            for a_i, agent in enumerate(env.world.agents):
                scenario.move_to_target(agent, targets[a_i][0], env.world)

            env.step([np.zeros(env.action_space[0].n)] * env.n)

        # ------------------- pheromone update ------------------- #
        # get RoI visited per agent. Not memory because the target was not necessarily visited
        trails_iter = [a.target_history for a in env.world.agents]
        lengths = np.zeros(nr_ants)
        for ant, trail in enumerate(trails_iter):
            for edge in zip(trail, trail[1:] + [trail[0]]):
                lengths[ant] += scenario.distances[edge]
        n_opt += 1
        path_length = np.sum(lengths)
        cost = combined_cost(path_length, np.amax(lengths))

        if cost < cost_opt:  # and scenario.global_visited >= scenario.num_landmarks:
            length_opt = path_length
            slowest_agent = np.max(lengths)
            cost_opt = cost
            trails_opt = trails_iter[:]
            n_opt = 1
        costs.append(cost)
        costs_mean = float(np.mean(costs[-n_mean:]))

        tau_lim[1] = 1/(evap_coeff*cost_opt)
        tau_lim[0] = 2*env.n*tau_lim[1]/nr_cities  # 2*tau_lim[1]/(nr_cities/env.n)

        if share_pheromone:
            # share a pheromone map (must also share tau[0] in _update_prob_matrix above)
            global_pheromone_update(tau[0], evap_coeff=evap_coeff, Q=q, fitness=cost,
                                    trails=trails_opt, tau_lim=tau_lim, ind_triu=ind_triu)
        else:
            for k in range(nr_tau):
                global_pheromone_update(tau[k], evap_coeff=evap_coeff, Q=q, fitness=cost_opt,
                                        trails=[trails_opt[k]], tau_lim=tau_lim, ind_triu=ind_triu)

        delta = (costs_mean - cost_opt) / cost_opt

        if (abs(delta) < tol and iters > n_mean) or n_opt > 2*n_mean:  # enforce a minimum number of iterations
            completed = True

        if completed:
            break
    env.reset()  # be nice, reset the env

    info = {'pheromone maps': np.power(tau, alpha),
            'desirability maps': np.power(eta, beta),
            'probability matrices': prob_matrix,
            'num iters': iters,
            'max iters reached': iters == max_iter,
            'optimal cost': cost_opt,
            'slowest agent': slowest_agent
            }

    tours = [trails_opt[k] for k in range(env.n)]

    return tours, length_opt, info
