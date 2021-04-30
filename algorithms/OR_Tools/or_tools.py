from ortools.constraint_solver import routing_enums_pb2, pywrapcp

from utility.vrp_utils import VRPDescriptor

runtime_default = 0


def or_tools_sa(env, scenario, phi=0., init_solution='NN', max_runtime=runtime_default, **kwargs):
    """Run OR Tools with Simulated Annealing."""
    vrp = VRPDescriptor(scenario, phi=phi)
    return apply_or_solver(vrp, init_solution, max_runtime, metaheuristic='SA')


def or_tools_ts(env, scenario, phi=0., init_solution='NN', max_runtime=runtime_default, **kwargs):
    """Run OR Tools with Tabu Search."""
    vrp = VRPDescriptor(scenario, phi=phi)
    return apply_or_solver(vrp, init_solution, max_runtime, metaheuristic='TS')


def or_tools_gls(env, scenario, phi=0., init_solution='NN', max_runtime=runtime_default, **kwargs):
    """Run OR Tools with Guided Local Search."""
    vrp = VRPDescriptor(scenario, phi=phi)
    return apply_or_solver(vrp, init_solution, max_runtime, metaheuristic='GLS')


def apply_or_solver(vrp, init_solution, max_runtime, metaheuristic, log_search=False):
    """
    Use the OR-Tools toolkit to find a solution to a vehicle routing problem.

    :param vrp: an initialized VRPDescriptor object, containing all necessary information
    :param init_solution: the type of initial solution to use. Default is 'NN' = PATH_CHEAPEST_ARC
    :param max_runtime: maximum time to run. If default value is used, time is scaled to problem size
    :param metaheuristic: metaheuristic to use. Defined by which callback is used
    :param log_search: turn output log on (verbose)
    """
    manager = pywrapcp.RoutingIndexManager(vrp.n_customers + 1, vrp.n_vehicles, vrp.depot)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(i, j):
        i_node = manager.IndexToNode(i)
        j_node = manager.IndexToNode(j)
        return vrp.distances[i_node][j_node] * vrp.scale_factor  # try it in meters - seems to work better

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add distance constraint
    dimension_name = 'Distance'
    routing.AddDimension(
        transit_callback_index,
        0,  # no slack
        int(vrp.max_distance * vrp.scale_factor),
        True,  # start cumul to zero
        dimension_name)
    distance_dimension = routing.GetDimensionOrDie(dimension_name)
    distance_dimension.SetGlobalSpanCostCoefficient(100)

    def demand_callback(i):
        i_node = manager.IndexToNode(i)
        return vrp.demand[i_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)

    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index, 0, [vrp.capacity for v in range(vrp.n_vehicles)], True, 'Capacity')

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    if init_solution == 'cw' or init_solution == 'CW':
        search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.SAVINGS
    elif init_solution == 'nearest_neighbour' or init_solution == 'NN':
        search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    elif init_solution == 'sweep':
        search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.SWEEP
    elif init_solution == 'christofides':
        search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.CHRISTOFIDES

    if metaheuristic == 'GLS':
        search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    elif metaheuristic == 'SA':
        search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.SIMULATED_ANNEALING
    elif metaheuristic == 'TS':
        search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.TABU_SEARCH

    if max_runtime == runtime_default:  # set time limit based on problem size
        if vrp.n_customers <= 50:
            max_runtime = int(50**3 / 2**8)
        else:
            max_runtime = int(vrp.n_customers**3 / 2**8)

    search_parameters.time_limit.seconds = max_runtime
    search_parameters.log_search = log_search

    solution_ = routing.SolveWithParameters(search_parameters)
    if not solution_:
        raise RuntimeError('No solution found with OR Tools!')

    solution, solution_agents = transform_solution(vrp, solution_, manager, routing)
    return {'solution': solution, 'info': {'GSCC': 100, 'Distance': vrp.max_distance * vrp.scale_factor,
                                           'status': routing.status(), 'agents': solution_agents}}


def transform_solution(vrp, solution, manager, routing):
    """Transform OR-Tools output solution into the correct shape."""
    paths = []
    n_agents = 0
    for v in range(vrp.n_vehicles):
        index = routing.Start(v)
        route = [manager.IndexToNode(index)]
        while not routing.IsEnd(index):
            index = solution.Value(routing.NextVar(index))
            route.append(manager.IndexToNode(index))
        paths.append(route)
        if len(route) > 2:
            n_agents += 1
    return paths, n_agents
