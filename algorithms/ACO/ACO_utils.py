import numpy as np
from typing import List


def global_pheromone_update(tau: np.ndarray, *, evap_coeff: float, Q:float, fitness: float,
                            trails: List[List[int]], tau_lim: List[int], **kwargs) -> None:
    if 'ind_triu' in kwargs:
        ind_triu = kwargs['ind_triu']
    else:
        n = tau.shape[0]
        ind_triu = np.triu_indices(n, k=1)

    # evaporate
    tau *= (1 - evap_coeff)

    # deposit pheromone on the trails
    del_tau = Q / fitness
    for trail in trails:
        for edge in zip(trail, trail[1:] + [trail[0]]):
            edge = (min(edge),max(edge))  # only upper triangle co-ords valid
            tau[edge] += del_tau

    # apply limits
    tau[ind_triu] = np.where(tau[ind_triu] > tau_lim[1], tau_lim[1], tau[ind_triu])
    tau[ind_triu] = np.where(tau[ind_triu] < tau_lim[0], tau_lim[0], tau[ind_triu])


def update_prob_matrix(prob_matrix: np.ndarray, *, tau: np.ndarray, eta_beta: np.ndarray, alpha: float, **kwargs) \
        -> None:
    """ Update the probability distribution for going to any city from any other city"""
    # note: need to normalise it later so that sum(prob_matrix[chosen]) == 1
    n = prob_matrix.shape[0]
    if 'ind_triu' in kwargs:
        ind_triu = kwargs['ind_triu']
    else:
        ind_triu = np.triu_indices(n, k=1)
    if 'ind_tril' in kwargs:
        ind_tril = kwargs['ind_tril']
    else:
        ind_tril = np.tril_indices(n, k=-1)

    # set probabilities to 0 for impossible edges
    if 'ind_invalid' in kwargs:
        ind_invalid = kwargs['ind_invalid']
        prob_matrix[ind_invalid] = 0

    temp1 = np.power(tau[ind_triu], alpha)
    prob_matrix[ind_triu] = np.multiply(temp1, eta_beta)

    # make symmetric
    prob_matrix[ind_tril] = 0
    prob_matrix += prob_matrix.T

    # exploit/greedy explore ratio
    # print("(policy exploit)/(policy explore) ratio", np.max(temp1)/np.mean(temp1))
    # print("(policy exploit) > heuristic ratio %.3f%%" % (np.sum(temp1 > eta_beta)/len(eta_beta)*100))
    # ind = np.argpartition(temp1, -n)[-n:]  # get n edges with highest probability in tau
    # print("(policy top)/heuristic ratio", np.sum(temp1[ind])/np.sum(eta_beta[ind]))


def get_targets_prob(agents, targets: List[List[int]], *, visited: List[bool], prob_matrix: np.ndarray,
                     origin_idx, n_nests=1, min_tour_length=1, rng=None) \
        -> List[List[int]]:
    """ Select targets using a probability distribution. .
    :param agents: list of agent objects. Must have the attribute "targets_history"
    :param targets: A list of previous targets for each agent. Might be empty or might be a list of targets
    :param visited: n x 1 list of targets of booleans for visited/not visited
    :param prob_matrix: symmetric (1 + n) x (1 + n) probability matrix for n points and 1 origin point
    :param origin_idx: index of split between nests/cities
    :param n_nests: number of agent nests
    :param min_tour_length: prevent agents from returning to the origin too soon because of a two-way feedback loop
    :param rng: a pseudo-random number generator object
    :return another list of targets
    """
    assert len(agents) == len(targets)
    if not rng:
        rng = np.random  # make a new pseudo random number generator
    nr_cities = len(visited)
    nr_agents = len(agents)
    cities = np.arange(0, nr_cities + n_nests)

    # Allow agents to return to the origin early, but ensure there is at least 1 active agent
    nr_at_origin = 0
    min_tour = np.inf  # get the minimum tour of all agents
    for i_agent, agent in enumerate(agents):
        min_tour = min(len(agent.target_history), min_tour)
        if len(targets[i_agent]) > 0:
            if targets[i_agent][0] == agent.spawn + origin_idx:  # is the agent's target the base?
                nr_at_origin += 1
    if nr_at_origin < (nr_agents - 1) and min_tour >= min_tour_length:
        visited.extend([False for _ in range(n_nests)])  # include the origins as an option -> deactivates an agent
    else:
        visited.extend([True for _ in range(n_nests)])   # there must be at least 1 active agent -> not origin
    # visited[0] = True  # all agents are always active

    for i_agent, sequence in enumerate(targets):
        if all(visited):
            break
        allocate = False
        if len(sequence) == 0:
            allocate = True  # no target, so get a new a target
        elif visited[sequence[0]] and not sequence[0] >= origin_idx:  # prevent reactivating "sleeping" agents
            allocate = True  # this target has been visited, so get a random new target
        if allocate:
            init = agents[i_agent].target_history[-1]
            _prob_vector = _get_prob_vector(init, prob_matrix[i_agent], visited)
            target = rng.choice(cities, p=_prob_vector)
            targets[i_agent] = [target]
            if target == agents[i_agent].spawn + origin_idx:  # if the target is the agent's origin
                nr_at_origin += 1
            if nr_at_origin >= (nr_agents - 1):  # prevent the last active agent from being deactivated
                for nest in range(n_nests):
                    visited[nest + origin_idx] = True
            # visited[target] = True  # prevent another agent going to the same target

    return targets


def _get_prob_vector(idx: int, prob_matrix: np.ndarray, visited) \
        -> np.ndarray:

    prob_vector = prob_matrix[idx, :].copy()  # == prob_matrix [:, idx] ... assume symmetric
    prob_vector[visited] = 0  # set visited probabilities to zero
    prob_vector[idx] = 0  # never choose previous position
    # prob_vector[-1] = 0  # never choose the last city, which is the origin city by convention
    prob_total = np.sum(prob_vector)
    if prob_total == 0:
        print("Error: empty probability vector in ACO. Returning non-normalized vector.")
    else:
        prob_vector = prob_vector / prob_total  # normalise so sum(prob_vector)==1
    return prob_vector


# ------------ main controller ------------ #
def _controller(error_pos, dynamics: str) -> np.ndarray:
    """ basic controllers for the different dynamics. Must pass raw errors. Should output a value between -1 and 1 """
    u = np.zeros(2)
    if dynamics == 'direct':
        # error_pos=[del_x, del_y], u = [del_x, del_y]
        # action should be directly proportional to the error
        u_norm = np.linalg.norm(error_pos)
        if u_norm < 1:
            u[:] = error_pos
        else:
            u[:] = error_pos/u_norm
    elif dynamics == 'velocity':
        # error_pos=[distance, bearing] , u = [linear velocity, angular velocity]
        # 2 separate proportional controllers for linear and angular velocity respectively
        v_max, w_max = 15, 3.15
        Kv, Kw = 2, 5
        u[0] = 1 if error_pos[0] > v_max else Kv*(error_pos[0]+0.1)/v_max
        u[1] = 1 if error_pos[1] > w_max else Kw*error_pos[1]/w_max
    elif dynamics == 'acceleration':
        # error_pos=[distance, bearing] , u = [linear acceleration, angular acceleration]
        # Use a pure pursuit controller with damping
        raise NotImplementedError
    else:
        raise NotImplementedError

    return u


def get_actions_sequences(sequences: List[List[int]], *, agents, points: np.array, dynamics: str, visited: List[bool]):
    """ Get actions using predefined targets for each agent. Calculate the errors directly from agents and cities"""
    assert len(agents) == len(sequences)
    nr_agents = len(agents)
    actions = np.zeros((nr_agents, 2))

    for i_agent, agent in enumerate(agents):
        if len(sequences[i_agent]) > 0:
            current = 0
            target = sequences[i_agent][current]
            while visited[target] and current < len(sequences[i_agent]):
                target = sequences[i_agent][current]  # note: this sequence must be updated in another function
                current += 1
            # calculate errors based on the dynamics
            delta = points[target, :] - agent.state.p_pos
            error_pos = np.zeros(2)
            if dynamics == 'direct':
                error_pos[:] = delta
            elif dynamics == 'velocity':
                error_pos[0] = (delta[0]**2 + delta[1]**2) ** 0.5
                bearing = np.arctan2(delta[1], delta[0]) - agent.state.ori_w
                bearing = bearing - 2 * np.pi if bearing > np.pi else bearing
                bearing = bearing + 2 * np.pi if bearing < -np.pi else bearing
                error_pos[1] = bearing
            else:
                raise NotImplementedError
            # get the controller
            actions[i_agent, :] = _controller(error_pos, dynamics)
    return actions
