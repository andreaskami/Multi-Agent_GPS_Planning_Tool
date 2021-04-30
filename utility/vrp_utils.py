import numpy as np

from algorithms.two_opt import two_opt_path
from utility.misc import get_path_distances


def cost(dists, z=0.5, q=0.5):
    """z: tradeoff between slowest agent and fairness. q: tradeoff between average & standard deviation btwn agents."""
    if z < 0 or z > 1 or q < 0 or q > 1:
        raise ValueError('Cost function: both z and q should be between 0 and 1')
    return z * dists['max'] + (1 - z) * (q * dists['avg'] + (1 - q) * dists['stdev'])


def render_path_planning_result(env, scenario, targets, path_lengths, im_path, absolute=False):
    """
    Gets the resulting path from a list of targets. Stores an image of the render at im_path file location.
    Also prints some output information. If absolute, expects a list of target coordinates, else landmark indices.
    """
    env.reset()
    scenario.set_distance_lut_diagonal(0)  # no distance to the same point

    length_total = path_lengths['total'] * scenario.topology.scale_factor
    length_slowest = path_lengths['max'] * scenario.topology.scale_factor
    length_average = path_lengths['avg'] * scenario.topology.scale_factor
    length_stdev = path_lengths['stdev'] * scenario.topology.scale_factor
    operation_time = length_slowest / scenario.agent_info['speed']  # flying time in seconds

    # print('Total path length: %.2fm' % length_total)
    # print('Longest agent distance: %.2fm' % length_slowest)
    # print('Average path length per agent: %.2fm' % length_average)
    # print('Standard deviation between agents: %.2fm' % length_stdev)
    # print('Total agent operation time: %.2fs' % operation_time)

    t_lengths = [len(target_list) for target_list in targets]
    for i in range(max(t_lengths)):
        for j, agent in enumerate(env.world.agents):
            if i < len(targets[j]):
                if absolute:
                    agent.state.p_pos = np.asarray(targets[j][i])

                else:
                    for loc in scenario.workaround_getter(agent.target_history[-1], targets[j][i]):
                        agent.state.p_pos = loc
                        env.step([np.zeros(env.action_space[0].n)] * env.n)
                    scenario.move_to_target(agent, targets[j][i], env.world)

        env.step([np.zeros(env.action_space[0].n)] * env.n)

    env.render(end_screen=True)
    env.store_images(im_path)

    return {'total': length_total, 'slowest': length_slowest, 'avg': length_average, 'stdev': length_stdev,
            'run_time': operation_time}


class VRPDescriptor:
    def __init__(self, scenario, capacity=1, demand=None, phi=0.85, speed_scale=1.0):
        """
        Initializes a vehicle routing problem (VRP) descriptor object. Assumes a single depot.
        Holds all relevant information, and can be used to calculate whether a certain route is valid.

        :param scenario: a complete instance of a Scenario object (see scenario.py)
        :param capacity: carrying capacity per vehicle. One number that is the same for each vehicle, default 1.
        :param demand: a list of demands per node (point of interest/customer). Default is 0.
        :param phi: a scaling factor used to force "fairness" on a system. 0 <= phi <= 1. The larger, the less fair.
                    This value is not used if parameter demand is used (is not None).
        :param speed_scale: scale factor for increasing / decreasing max_distance from agent info.
        """
        self.depot = scenario.num_landmarks
        self.n_vehicles = scenario.num_agents
        self.n_customers = scenario.num_landmarks

        self.capacity = capacity
        self.phi = phi
        if demand is None:
            if phi < 0 or phi > 1:
                raise ValueError('Phi constant should satisfy 0 <= phi <= 1')
            demand_per_node = (self.n_vehicles / self.n_customers) * phi
            self.demand = [demand_per_node for _ in range(self.n_customers)] + [0]  # depot has a demand of 0
        else:
            self.demand = demand

        self.distances = scenario.distances
        self.max_distance = scenario.agent_info['distlimit'] * speed_scale
        self.scale_factor = scenario.topology.scale_factor  # for conversion to meters

    def __len__(self):
        return self.n_customers  # amount of customers

    def valid_route(self, route):
        """
        Checks whether a given (single) route is valid with regard to max_distance, capacity and demand.
        Also returns route length as a second argument (since it is calculated anyway...).
        """
        if len(route) == 0:
            return True, 0  # an empty route is a valid one and has a length of 0
        route_demand = 0
        dist = self.distances[route[0]][self.depot] + self.distances[self.depot][route[-1]]
        for n, node in enumerate(route):
            route_demand += self.demand[node]
            if n != 0:
                dist += self.distances[node][route[n - 1]]
        return route_demand <= self.capacity and dist <= self.max_distance, dist

    def route_demand(self, route):
        """Calculates the total demand for a given route."""
        route_demand = 0
        for node in route:
            route_demand += self.demand[node]
        return route_demand

    def wrap_routes(self, routes):
        """
        Makes sure that all routes in the routes object start and end with the depot value.
        Empty routes are assigned with only the depot by this function
        """
        for r, route in enumerate(routes):
            if len(route) == 0:
                routes[r] = [self.depot]
            else:
                if route[0] != self.depot and route[-1] != self.depot:
                    routes[r] = [self.depot] + route + [self.depot]
                elif route[0] == self.depot and route[-1] != self.depot:
                    routes[r] = route + [self.depot]
                elif route[0] != self.depot and route[-1] == self.depot:
                    routes[r] = [self.depot] + route

    def match_route_lengths_and_wrap(self, routes):
        """Matches route lengths to the number of agents, either by adding empty routes or combining shortest paths"""
        if len(routes) > self.n_vehicles:
            routes = self.combine_shortest_routes(routes)
        elif len(routes) < self.n_vehicles:
            for a in range(len(routes), self.n_vehicles):
                routes.append([])  # extra agents do nothing
        self.wrap_routes(routes)
        return routes

    def combine_shortest_routes(self, routes):
        """
        Combines shortest paths until the number of routes matches the number of agents. Assumes depot is not included.
        2-opt is used to optimize final path.
        """
        lengths = get_path_distances(routes, self.distances)['list']

        while len(routes) > self.n_vehicles:
            idxs = np.argpartition(lengths, 2)[0:2]  # get indices of two shortest routes
            new_path = two_opt_path(routes[idxs[0]] + routes[idxs[1]], self.depot, self.distances)[0]
            # new_path = routes[idxs[0]] + routes[idxs[1]]

            if not self.valid_route(new_path)[0] and np.sum(lengths[idxs]) > self.max_distance:
                print('Warning: a combination of two routes does not satisfy robot battery constraints!',
                      'Try increasing the number of agents for this scenario.')

            routes.append(new_path)  # insert new path
            routes.pop(max(idxs))  # delete original paths
            routes.pop(min(idxs))

            lengths = np.append(lengths, np.sum(lengths[idxs]))  # update length of new path
            lengths = np.delete(lengths, idxs)  # delete invalid lenghts
        return routes
