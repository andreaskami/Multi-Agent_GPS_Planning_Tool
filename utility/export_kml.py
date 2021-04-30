import os
import simplekml


def export_to_kml(scenario, world, routes, file_path, include_landmarks_and_spawns=True):
    """Exports a solution to paths in a .kml file."""
    paths = scenario.get_target_coordinate_lists(world, routes)
    paths = scenario.topology.get_real_world_coordinates(paths)

    colors = [agent.color for agent in world.agents]

    if include_landmarks_and_spawns:
        points = scenario.get_locations(world)
        points = scenario.topology.get_real_world_coordinates(points)
        landmarks, spawns = points[:-1], points[-1:]
    else:
        landmarks, spawns = None, None

    create_kml_file(paths, file_path, colors, landmarks, spawns)


def create_kml_file(routes, file_path, route_colors=None, landmarks=None, spawns=None):
    """Generate kml file. Note: points should be lon, lat formatted."""

    kml = simplekml.Kml()

    for r, route in enumerate(routes):
        ls = kml.newlinestring(name='agent_route_%d' % r)
        ls.coords = route

        if route_colors is not None:
            color = (route_colors[r] * 255).astype(int)
            ls.style.linestyle.color = simplekml.Color.rgb(*color)
        ls.style.linestyle.width = 5

    if landmarks is not None:
        landmark_style = simplekml.Style()
        landmark_style.iconstyle.color = simplekml.Color.white  # icon color - does not seem to be working?
        landmark_style.iconstyle.scale = 0.5  # small icons
        landmark_style.labelstyle.scale = 0.0  # hide text label

        for i, landmark in enumerate(landmarks):
            pt = kml.newpoint(name='%d' % i)
            pt.coords = [landmark]
            pt.style = landmark_style

    if spawns is not None:
        spawn_style = simplekml.Style()
        spawn_style.iconstyle.color = simplekml.Color.red  # red spawn

        for s, spawn in enumerate(spawns):
            pt = kml.newpoint(name='spawn_%d' % s)
            pt.coords = [spawn]
            pt.style = spawn_style

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    kml.save(file_path)
