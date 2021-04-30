import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from algorithms.DRL_VRP.model import DRL4TSP
from algorithms.DRL_VRP.tasks.vrp import VehicleRoutingDataset, reward

from utility.vrp_utils import VRPDescriptor
from utility.misc import normalize_coordinates
from utility.file_io import get_project_root


def drlvrp_inference(env, scenario, model=None, settings=None, phi=1., **kwargs):
    """
    Runs inference on a pre-trained model for Deep Reinforcement Learning for the VRP. Uses PyTorch.
    Deep Reinforcement Learning for Solving the Vehicle Routing Problem - M. Nazari et al., 2018
    Pytorch adaptation of the method by mveres01 on GitHub: https://github.com/mveres01/pytorch-drl4vrp

    Uses greedy approach. TODO add beam search as well?

    :param env: an instance of the MultiAgentEnv class
    :param scenario: an instance of the Scenario class (see scenario.py)
    :param model: file path to trained model. Given folder should contain a file named 'actor.pt', with a torch nn mod.
    :param settings: model settings. If None, uses default.
                     Should contain static_size, dynamic_size, hidden_size, num_layers and dropout.
    :param phi: scaling factor to generate customer demands (see vrp_utils.py)
    """

    if model is None:
        model = _get_closest_model(scenario.num_landmarks)

    if settings is None:
        static_size = 2  # (x, y)
        dynamic_size = 2  # (load, demand)
        hidden_size = 128
        num_layers = 1
        dropout = 0.1
    else:
        static_size = settings['static_size']
        dynamic_size = settings['dynamic_size']
        hidden_size = settings['hidden_size']
        num_layers = settings['num_layers']
        dropout = settings['dropout']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if phi <= 0:
        raise ValueError('Phi should be larger than 0 for inference on DRL for VRP.')
    vrp = VRPDescriptor(scenario, phi=phi)

    inference_instance = VRInferenceDataset(env, scenario, max_load=1, max_demand=1,
                                            demand=[vrp.demand[-1]] + vrp.demand[:-1])
    inference_loader = DataLoader(inference_instance)

    actor = DRL4TSP(static_size, dynamic_size, hidden_size,
                    inference_instance.update_dynamic,
                    inference_instance.update_mask,
                    num_layers=num_layers,
                    dropout=dropout).to(device)

    checkpoint = os.path.join(model, 'actor.pt')
    actor.load_state_dict(torch.load(checkpoint, device))

    actor.eval()

    instance = next(iter(inference_loader))  # get a single instance from the dataset

    static, dynamic, x0 = instance

    static = static.to(device)
    dynamic = dynamic.to(device)
    x0 = x0.to(device) if len(x0) > 0 else None

    with torch.no_grad():
        tour_indices, _ = actor.forward(static, dynamic, x0)

    score = reward(static, tour_indices)
    solution, n_agents = _transform_output(tour_indices, vrp)
    return {'solution': solution, 'info': {'network_score': score.item(), 'agents': n_agents}}


def _transform_output(tensor_indices, vrp):
    """Gets solution for output from tensor. Also returns the number of routes found by the network."""
    tensor_indices -= 1  # map to indexing of scenario
    paths_combined = tensor_indices.squeeze().tolist()

    solution = []
    prev_idx = 0
    for idx, loc in enumerate(paths_combined):
        if loc < 0:  # depot is at -1 right now
            solution.append(paths_combined[prev_idx:idx])
            prev_idx = idx + 1
    n_agents = len(solution)
    return vrp.match_route_lengths_and_wrap(solution), n_agents


def _get_closest_model(n_nodes):
    """Find the closest model for the current problem size from the available models in ./VRP/DRL_VRP/models/"""
    model_folder = os.path.join(get_project_root(), 'VRP', 'DRL_VRP', 'models')
    model_dirnames = [name for name in os.listdir(model_folder) if os.path.isdir(os.path.join(model_folder, name))]
    models = [int(s.strip('vrp')) for s in model_dirnames]  # get list of integers that represent the available models
    closest = models[0]
    best_err = abs(n_nodes - models[0])
    for model in models[1:]:
        if abs(n_nodes - model) < best_err:  # this model is a better match
            best_err = abs(n_nodes - model)
            closest = model
    return os.path.join(model_folder, 'vrp' + str(closest))


class VRInferenceDataset(VehicleRoutingDataset):
    """VehicleRoutingDataset wrapper class to transform a scenario to the correct format."""
    def __init__(self, env, scenario, max_load=30, max_demand=9, demand=None):
        super(VRInferenceDataset, self).__init__(1, scenario.num_landmarks, max_load, max_demand)
        if max_load < max_demand:
            raise ValueError(':param max_load: must be > max_demand')

        if demand is None:  # 0 for depot, other random in the range 1 up to including 9
            self.demand = [0] + [np.random.randint(1, max_demand + 1) for _ in range(scenario.num_landmarks)]
        else:
            self.demand = demand

        locations = self._get_normalized_locs(env, scenario)
        self.static = torch.FloatTensor(locations)

        dynamic_shape = (self.num_samples, 1, scenario.num_landmarks + 1)
        loads = torch.full(dynamic_shape, 1.)

        demands_ = [[[dem for dem in self.demand]]]
        demands = torch.FloatTensor(demands_)
        demands = demands / float(max_load)

        if demands[:, 0, 0] != 0:
            demands[:, 0, 0] = 0  # depot demand should be 0

        # dynamic out = tuple with all loads and demands
        self.dynamic = torch.tensor(np.concatenate((loads, demands), axis=1))

    @staticmethod
    def _get_normalized_locs(env, scenario):
        locs = scenario.get_locations(env.world)
        locs, _ = normalize_coordinates(locs, np.asarray([(-1, -1), (1, 1)]), np.asarray([(0, 0), (1, 1)]))
        locs.insert(0, locs.pop())  # move depot to front
        return [[[loc[0] for loc in locs], [loc[1] for loc in locs]]]


