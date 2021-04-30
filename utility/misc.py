from multiagent.core import Entity
import numpy as np


def hash_solution(routes):
    """Returns a hash representation value for the current set of paths"""
    tp = tuple([tuple(route) for route in routes])
    return hash(tp)


def distance(loc, dest):
    """Returns the Euclidian distance between two entities or point coordinates"""
    location = loc.state.p_pos if isinstance(loc, Entity) else np.array(loc)
    destination = dest.state.p_pos if isinstance(dest, Entity) else np.array(dest)
    return np.linalg.norm(location - destination)


def distance_lon_lat_meter(loc, dest):
    """Returns the distance between two lon-lat points in meters"""
    radius = 6.371230 * 1e6  # Mean radius of the earth in m - average distance from center to surface
    dist_rad = np.radians(np.asarray(loc) - np.asarray(dest))
    a = (np.sin(dist_rad[1] * 0.5) * np.sin(dist_rad[1] * 0.5) +
         np.cos(np.radians(loc[1])) * np.cos(np.radians(dest[1])) *
         np.sin(dist_rad[0] * 0.5) * np.sin(dist_rad[0] * 0.5))
    return radius * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def normalize_coordinates(locations, input_bounds, target_bounds, keep_aspect_ratio=False, aspect_max=False):
    """
    Maps a list of locations to another scale and offset as specified by the target_bounds.
    If keeping aspect ratio, the scale factor is either based on the max axis or min axis (determined by aspect_max).
    """
    if input_bounds is None:
        return locations  # No input bounds calculated yet. Incorrect use of function. Return original locations
    if target_bounds is None:
        return locations  # No target bounds provided. Incorrect use of function. Return original locations
    if keep_aspect_ratio:
        if aspect_max:
            scale_factor = np.max((target_bounds[1] - target_bounds[0]) /
                                  (input_bounds[1] - input_bounds[0]))
        else:
            scale_factor = np.min((target_bounds[1] - target_bounds[0]) /
                                  (input_bounds[1] - input_bounds[0]))
    else:
        scale_factor = np.array((target_bounds[1] - target_bounds[0]) /
                                (input_bounds[1] - input_bounds[0]))
    mapped_pts = []
    for pt in locations:
        mapped_pt = target_bounds[0] + (np.asarray(pt) - input_bounds[0]) * scale_factor
        if type(pt) is tuple:
            mapped_pts.append(tuple(mapped_pt))
        else:
            mapped_pts.append(list(mapped_pt))
    return mapped_pts, scale_factor


def get_path_distances(targets, distances, absolute=False, ignore_zeros=True):
    """
    Returns lengths of all found paths, the longest of these paths and the total, avg and stdev of all paths
    Distances should have a 0 as the diagonal value.
    If absolute, the function expects absolute coordinates. Otherwise, target landmark indices.
    """
    dist = np.zeros(len(targets))
    for a, target_list in enumerate(targets):
        if len(target_list) <= 1:
            continue
        for i, target in enumerate(target_list):
            if i == 0:
                continue
            same_loc = (target == [target_list[i - 1]]).all() if isinstance(target, np.ndarray) \
                else target == [target_list[i - 1]]
            if not same_loc:  # do not add distance if staying in the same location
                if absolute:
                    dist[a] += distance(target, target_list[i - 1])
                else:
                    dist[a] += distances[target][target_list[i - 1]]
    # Ignore zero values in calculating mean and standard deviation if ignore_zeros is True
    avg = np.true_divide(dist.sum(), (dist != 0).sum()) if ignore_zeros else dist.mean()
    stdev = dist[np.invert(np.isclose(dist, 0))].std() if ignore_zeros else dist.std()
    return {'list': dist, 'max': np.max(dist), 'total': dist.sum(), 'avg': avg, 'stdev': stdev}


def single_path_distance(path, distances):
    """Returns lengths of a single path (list of landmark indices)."""
    dist = 0
    for n, node in enumerate(path[:-1]):
        dist += distances[path[n]][path[n + 1]]
    return dist


def get_polar_angle(origin, point):
    """Both point and origin should be numpy arrays of coordinate sets"""
    point_ = point - origin  # translate point by origin
    return np.arctan2(point_[1], point_[0])


def polar_to_cartesian(origin, angle, magnitude):
    """Convert a polar angle and magnitude to x, y coordinates."""
    point_ = np.asarray([magnitude * np.cos(angle), magnitude * np.sin(angle)])
    return point_ + origin


def get_one_hot(nd_array, n):
    """Transform a numpy array of indices into an array of one-hot encodings of size n."""
    one_hot = np.zeros((nd_array.size, n))
    one_hot[np.arange(nd_array.size), nd_array] = 1.
    return one_hot


def generate_random_polygon(n, ctr_loc=(.0, .0), avg_radius=.9, min_radius=.4, max_radius=1.0,
                            irregularity=0, spikeyness=0):
    """
    Generates a random polygon according to the various variables that can be passed as arguments.
    Code based on code by Mike Ounsworth on StackOverflow:
    https://stackoverflow.com/questions/8997099/algorithm-to-generate-random-2d-polygon

    :param n: number of points to add to the polygon
    :param ctr_loc: coordinates of the center location of the polygon
    :param avg_radius: average radius of the polygon
    :param min_radius: minimum radius of the polygon (smallest allowed distance from point to center)
    :param max_radius: maximum radius of the polygon
    :param irregularity: value between 0 and 1 - higher value means more irregular spacing of points around the center
    :param spikeyness: value between 0 and 1 - higher value means larger devation from the average radius
    :return: a list of tuple coordinates, one tuple for each pair of points
    """
    irregularity = np.clip(irregularity, 0, 1) * 2 * np.pi / n
    spikeyness = np.clip(spikeyness, 0, 1) * avg_radius
    # generate n angle steps
    lower = (2 * np.pi / n) - irregularity
    upper = (2 * np.pi / n) + irregularity
    angle_steps = np.random.uniform(lower, upper, n) * 2 * np.pi
    total_angle = np.sum(angle_steps)
    # normalize the steps so that point 0 and point n+1 are the same
    k = total_angle / (2 * np.pi)
    angle_steps = angle_steps / k
    points = []
    current_angle = np.random.rand() * 2 * np.pi
    for i in range(n):
        r_i = np.clip(np.random.normal(avg_radius, spikeyness), min_radius, max_radius)
        x = ctr_loc[0] + r_i * np.cos(current_angle)
        y = ctr_loc[1] + r_i * np.sin(current_angle)
        points.append((x, y))
        current_angle += angle_steps[i]
    return points
