import xml.etree.ElementTree as et
import numpy as np
import sys

from csv import DictWriter

from utility.misc import distance_lon_lat_meter, normalize_coordinates


class KMLTopology:
    """
    Requires the KML file to be in the following format:
    A single place folder, with only polygons and placemarks.
    Agent locations (placemarks) should contain the lowercase word 'agent' in their name.
    The world boundary (polygon) should contain the lowercase word 'boundary' in its name.
    Polygon obstacles and the boundary will be set to 'hard' if their name contains 'hard'.

    For more information, see the google_earth_guide in the project root folder.
    """
    def __init__(self, path, world_limits=None):
        self.target_bounds = world_limits
        if self.target_bounds is None:
            self.target_bounds = np.asarray([(-1, -1), (1, 1)])  # Default world limits
        else:
            self.target_bounds = np.asarray(self.target_bounds)

        self.boundary = []
        self.walls = []
        self.landmarks = []
        self.agents = []

        # values linking to boundary .csv
        self.boundary_location = ''
        self.boundary_output = []

        self.names = {'boundary': None, 'walls': [], 'landmarks': [], 'agents': [], 'spawns': []}

        self.scale_factor = 1
        self.bounds = None
        self.bounds_meters = None

        try:
            kml_file = et.parse(path)
        except FileNotFoundError:
            print('The file ' + path + ' does not exist. Exiting.')
            sys.exit()

        nmsp = '{http://www.opengis.net/kml/2.2}'

        # loop through all locations in the .kml file
        for pm in kml_file.iterfind('.//{0}Placemark'.format(nmsp)):
            location_name = pm.find('{0}name'.format(nmsp)).text
            pts = pm.findtext('{0}Polygon/{0}outerBoundaryIs/{0}LinearRing/{0}coordinates'.format(nmsp))
            if pts is None:
                # Landmark found instead of polygon
                c = pm.findtext('{0}Point/{0}coordinates'.format(nmsp))
                if c is None:
                    continue  # invalid Geo object found
                if location_name.find('agent') != -1:
                    self.agents.append(eval(c)[0:2])  # Agent location found
                    self.names['spawns'].append(location_name)
                else:
                    self.landmarks.append(eval(c)[0:2])  # Landmark location found
                    self.names['landmarks'].append(location_name)
            else:
                c = [eval(pt)[0:2] for pt in pts.strip().split()]
                if location_name.find('boundary') != -1:
                    self.bounds, self.bounds_meters = self.get_input_bounds(c)  # Set input bounds
                    self.boundary = c  # Boundary polygon object found
                    self.names['boundary'] = location_name
                    # store boundary pts in output csv (geofence).
                    for fid, pt in enumerate(self.boundary):
                        self.boundary_output.append({'fid': fid + 1, 'lat': pt[1], 'lon': pt[0]})
                    self.boundary_location = path[:-4] + '_geofence.csv'
                    self.geofence_to_csv(self.boundary_output, self.boundary_location)
                else:
                    self.walls.append(c)  # Other polygon object found
                    self.names['walls'].append(location_name)

    def get_world_locations(self):
        """Returns normalized locations for boundary, walls, landmarks and agents. Stores scale factor as well."""
        self.boundary, _ = normalize_coordinates(self.boundary, self.bounds, self.bounds_meters)
        # Store second scale factor to convert distances to meters later
        self.boundary, self.scale_factor = normalize_coordinates(self.boundary, self.bounds_meters,
                                                                 self.target_bounds, keep_aspect_ratio=True)
        self.landmarks, _ = normalize_coordinates(self.landmarks, self.bounds, self.bounds_meters)
        self.landmarks, _ = normalize_coordinates(self.landmarks, self.bounds_meters,
                                                  self.target_bounds, keep_aspect_ratio=True)
        self.agents, _ = normalize_coordinates(self.agents, self.bounds, self.bounds_meters)
        self.agents, _ = normalize_coordinates(self.agents, self.bounds_meters,
                                               self.target_bounds, keep_aspect_ratio=True)
        for i, wall in enumerate(self.walls):
            wall, _ = normalize_coordinates(wall, self.bounds, self.bounds_meters)
            self.walls[i], _ = normalize_coordinates(wall, self.bounds_meters,
                                                     self.target_bounds, keep_aspect_ratio=True)
        return self.boundary, self.walls, self.landmarks, self.agents

    @staticmethod
    def get_input_bounds(boundary):
        """Get min/max values for the imported topology, and get a converted set of bounds to metres as well"""
        bounds = np.array((np.min(np.array(boundary), axis=0), np.max(np.array(boundary), axis=0)))
        width = distance_lon_lat_meter(bounds[0, :], [bounds[1, 0], bounds[0, 1]])
        height = distance_lon_lat_meter(bounds[0, :], [bounds[0, 0], bounds[1, 1]])
        bounds_meters = np.array(([-width/2, -height/2], [width/2, height/2]))
        return bounds, bounds_meters

    @staticmethod
    def geofence_to_csv(boundary_list, file_path):
        """Writes the geofence (real world boundary points) of an environment to the specified .csv file."""
        if file_path[-4:] != '.csv':
            file_path += '.csv'

        with open(file_path, 'w', newline='') as geofence_file:
            headers = ['fid', 'lat', 'lon']
            writer = DictWriter(geofence_file, fieldnames=headers)
            writer.writeheader()
            for row in boundary_list:
                writer.writerow(row)



