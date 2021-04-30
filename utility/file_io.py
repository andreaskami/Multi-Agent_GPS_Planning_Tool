from multiagent.core import World
from scenario import Scenario
from topology import Topology

from csv import DictWriter, DictReader, writer, reader
from pathlib import Path
import os
import shutil


def load_class_objects(file_path):
    """
    Load topology, scenario and world objects from file.
    See load_environment() and load_topology() for naming conventions.
    """
    top = Topology()
    top.load_topology(file_path)
    sc = Scenario(top)
    wrld = World()
    sc.load_environment(wrld, file_path)
    return top, sc, wrld


def store_class_objects(top, sc, wrld, file_path):
    """
    Store topology, scenario and world objects in files.
    See store_environment() and store_topology() for naming conventions.
    """
    top.store_topology(file_path)
    sc.store_environment(wrld, file_path)


def save_results_csv(results, file_path):
    """Results should be a non-empty list of dict result entries (one for each row), with matching headers"""
    if file_path[-4:] != '.csv':
        file_path += '.csv'

    with open(file_path, 'w', newline='') as output_file:
        headers = list(results[0].keys())
        res_writer = DictWriter(output_file, fieldnames=headers)
        res_writer.writeheader()
        for row in results:
            res_writer.writerow(row)


def read_results_csv(file_path):
    """Reads .csv file and stores it in a list of dict entries, like the one used in save_results_csv."""
    if file_path[-4:] != '.csv':
        file_path += '.csv'

    results = []
    try:
        with open(file_path, 'r', newline='') as results_file:
            res_reader = DictReader(results_file)
            for row in res_reader:
                results.append(row)
    except FileNotFoundError:
        results = []

    return results


def save_csv_solution(solution, file_path):
    """Stores a path solution to .csv."""
    if file_path[-4:] != '.csv':
        file_path += '.csv'

    with open(file_path, 'w') as solution_file:
        path_writer = writer(solution_file)
        path_writer.writerows(solution)


def read_csv_solution(file_path):
    """Reads a path from a .csv file (formatted for wadl right now), and returns a coordinate list."""
    if file_path[-4:] != '.csv':
        file_path += '.csv'

    solution = []
    with open(file_path, 'r') as solution_file:
        path_reader = reader(solution_file)
        for row in path_reader:
            solution.append((float(row[1]), float(row[0])))  # reverse for lat-lon instead of lon-lat
    return solution


def clear_output_folder(output_dir, exception=''):
    """Will attempt to clear all contents of the provided folder, except if the filename is equal to the exception"""
    for filename in os.listdir(output_dir):
        if filename == exception:
            continue
        file_path = os.path.join(output_dir, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def get_top_dir_name(output_dir):
    """Gets the name of the top directory in a specified folder. Currently used in POPCORN for evaluating output."""
    for filename in os.listdir(output_dir):
        file_path = os.path.join(output_dir, filename)
        if os.path.isdir(file_path):
            return filename
    return ''


def get_project_root():
    """Returns the absolute path to the project root folder."""
    return Path(__file__).parent.parent.__str__()
