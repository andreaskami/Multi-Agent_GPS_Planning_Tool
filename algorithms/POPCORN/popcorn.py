from wadl.survey import Survey
from wadl.lib.route import RouteParameters
from wadl.solver.solver import SolverParameters

from utility.file_io import read_csv_solution, get_top_dir_name

import os


def popcorn_cpp(env, scenario, step_size=None, clear_output=True, view=False, plot=False, **kwargs):
    """
    Performs coverage path planning based on Satifyability Modulo Theory (SMT).
    Needs Python 3.8 or higher to use the wadl library.

    Multidrone Aerial Surveys of Penguin Colonies in Antarctica - K. Shah et al., 2020

    :param env: an instance of the MultiAgentEnv class (currently not used by this function)
    :param scenario: an instance of the Scenario class (see scenario.py)
    :param step_size: grid step size for CPP, in meters
    :param clear_output: if True, clears all contents in the output folder afterwards (except the wadl.log log file)
    :param view: if True, shows the geofence beforehand in a plot
    :param plot: if True, shows the planned path(s) in a plot after running (via matplotlib.pyplot).
    :return: found solution (in relative environment coordinates).
    """

    if scenario.topology.flags['ground_flag']:
        raise ValueError('CPP algorithm will disregard obstacles. Use aerial agents instead!')

    output_dir = 'output'  # TODO find way to change output folder. Make issue in wadl git?

    if step_size is None:  # TODO find a way to match Scenario grid exactly
        step_size = scenario.topology.flags['grid_size_l'] * scenario.topology.scale_factor  # convert to meters

    spawn_loc = scenario.topology.get_real_world_coordinates(scenario.topology.spawns)
    key_points = {'spawn_%d' % i: spawn_loc[i][::-1] for i in range(len(scenario.topology.spawns))}

    survey = Survey(output_dir)
    survey.setKeyPoints(key_points)

    # set route parameters
    route_params = RouteParameters()
    route_params['limit'] = scenario.agent_info['life']
    route_params['speed'] = scenario.agent_info['speed']
    route_params['xfer_ascend'] = scenario.agent_info['vspeed']
    route_params['xfer_descend'] = scenario.agent_info['vspeed']

    survey.addTask(scenario.topology.boundary_file_loc,  # 'top_dataset/' +
                   step=step_size, home=list(key_points.keys()), routeParameters=route_params)

    # set solver parameters
    solver_params = SolverParameters()
    survey.setSolverParamters(solver_params)

    if view:
        survey.view()

    survey.plan(showPlot=plot)

    res_dir = get_top_dir_name(output_dir)
    n_routes = int(res_dir[res_dir.rfind('r') + 1:])
    paths = []
    for p in range(n_routes):
        path_file = os.path.join(output_dir, res_dir, 'routes', str(p))
        path_rel_coor = scenario.topology.get_environment_coordinates(read_csv_solution(path_file))
        paths.append(path_rel_coor)

    if len(paths) < scenario.num_agents:
        paths += [[scenario.topology.spawns[0]] for _ in range(scenario.num_agents - len(paths))]

    if clear_output:
        del survey, route_params, solver_params  # get ready for clearing output

    return {'solution': paths, 'info': {'agents': n_routes}}
