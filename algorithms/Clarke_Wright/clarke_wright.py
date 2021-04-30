from utility.vrp_utils import VRPDescriptor
from copy import deepcopy


def clarke_wright(env, scenario, cw_type='parallel', capacity=1, phi=0., speed_scale=1.0, **kwargs):
    """
    Creates an instance of the Clarke-Wright (CW) class. Next, the CW savings method is applied to generate a number
    of paths that are a solution to the vehicle routing problem, either through the sequential or parallel approach.
    The routes are then changed to match the number of agents, and wrapped so that they all start and end at the depot.

    :param env: an instance of the MultiAgentEnv class (currently not used by this function)
    :param scenario: an instance of the Scenario class (see scenario.py)
    :param cw_type: either 'parallel' or 'sequential' - sequential method is not supported yet.
    :param capacity: the weight capacity for each vehicle - a single value for all vehicles, or a list of diff values
    :param phi: scaling factor to generate customer demands (see vrp_utils.py).
    :param speed_scale: scaling factor for max_distance (see vrp_utils.py).
    """
    vrp = VRPDescriptor(scenario, capacity=capacity, phi=phi, speed_scale=speed_scale)
    cw_solver = CW(vrp)

    if cw_type == 'parallel':
        paths = cw_solver.cw_parallel()
    elif cw_type == 'sequential':
        paths = cw_solver.cw_sequential()
    else:
        raise ValueError('Unsupported cw_type variable.')

    paths = vrp.match_route_lengths_and_wrap(paths)
    return {'solution': paths, 'info': {'cw': cw_solver, 'type': cw_type,
                                        'agents': len(cw_solver.most_recent_solution)}}


class CW(object):
    def __init__(self, vrp):
        """
        Class with all necessary information to apply the Clarke-Wright savings algorithm.
        Scheduling of Vehicles from a Central Depot to a Number of Delivery Points - G. Clarke & J.W. Wright, 1964

        :param vrp: an instance of the VRPDescriptor class, holding all VRP information
        """
        self.vrp = vrp

        self.pairs = self._get_savings_list()  # calculate ordered list of pairs, sorted by savings

        self.most_recent_solution = None

    def cw_parallel(self):
        routes = [list(self.pairs[0])]  # top pair is an initial route - at least one vehicle

        # perform a pass through the ordered list of pairs
        for p, pair in enumerate(self.pairs):

            add_on = None
            occurances = 0
            for r, route in enumerate(routes):
                common_nodes = [node in route for node in pair]
                occurances += sum(common_nodes)
                if sum(common_nodes) == 2:  # both of these nodes are in the current route - immediate discard
                    add_on = None  # delete any earlier defined pair
                elif sum(common_nodes) == 1:  # one node overlaps - possible match
                    overlap = common_nodes.index(True)  # find index of pair that overlaps
                    if pair[overlap] == route[0]:  # possible addition at begin of route
                        route_ = [pair[not overlap]] + route
                        if self.vrp.valid_route(route_)[0] and add_on is None:
                            add_on = (r, 0, pair[not overlap])
                        continue  # keep searching
                    elif pair[overlap] == route[-1]:  # possible addition at end of route
                        route_ = route + [pair[not overlap]]
                        if self.vrp.valid_route(route_)[0] and add_on is None:
                            add_on = (r, len(route), pair[not overlap])
                        continue  # keep searching
                    else:
                        add_on = None  # discard pair - node part of interior of a route so no longer to be considered

            if occurances == 0:
                routes.append(list(pair))
            if occurances == 1 and add_on is not None:
                routes[add_on[0]].insert(add_on[1], add_on[2])

        self.most_recent_solution = deepcopy(routes)  # store in solution buffer
        return routes

    def cw_sequential(self):
        # TODO add sequential CW algorithm - low priority
        return [[] for _ in self.vrp.n_vehicles]

    def _get_savings_list(self):
        savings = {}
        for i in range(self.vrp.n_customers):
            for j in range(self.vrp.n_customers):
                if i > j:
                    savings[(i, j)] = self._calc_savings(self.vrp.depot, i, j, self.vrp.distances)
        return sorted(savings, key=savings.get, reverse=True)

    @staticmethod
    def _calc_savings(depot_idx, loc_0, loc_1, distances):
        dist_0 = distances[depot_idx][loc_0]
        dist_1 = distances[loc_1][depot_idx]
        dist_combined = distances[loc_0][loc_1]
        return dist_0 + dist_1 - dist_combined
