import csv
import unittest
from algorithms.ACO import aco
from optimization_interface.user_info import User
from optimization_interface.optimization import Optimization

# Download the Bigtable library: pip install google-cloud-bigtable
# Download the service account, JSON file from the simulation github (AdminBT.json)
# Set environment variable, GOOGLE_APPLICATION_CREDENTIALS:
# GOOGLE_APPLICATION_CREDENTIALS="/path to/AdminBT.json"


class Analysis(unittest.TestCase):
    def setUp(self):
        self.net_xml_path = '../../../optimization_interface/DCC.net.xml'
        self.start_mode = 'walking'
        self.ant_num = [2000]
        self.station_num = [20]
        self.energy_rate = [1]
        self.simulation_time = [20000]

        self.episodes = [500, 1000, 1500, 2000]
        self.iteration = 1
        self.db_path = '../../../optimization_interface/test_new.db'
        self.user = User(60, True, 0, 20)

    def test_run_aco(self):
        with open('../../../optimization_interface/od_pairs/od_pairs_500_new.csv', 'r') as file:
            reader = csv.reader(file)
            od_pairs = [tuple(row) for row in reader]

        test_size = len(od_pairs)
        all_aco_exe_time_costs = []
        all_aco_time_costs = []

        with open('results/ACO-noecar.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Experiment ID', 'Number of Ants', 'Simulation Time', 'Station Number', 'Initial Energy', 'Travel Time Cost (seconds)',
                             'Execution Time (seconds)', 'Path', 'Find'])


            for i in range(test_size):
                source_edge, target_edge = od_pairs[i]
                optimizer_interface = Optimization(self.net_xml_path, self.user, self.db_path, self.simulation_time[0], self.station_num[0], source_edge, target_edge)
                graph = optimizer_interface.new_graph
                if graph is None:
                    writer.writerow([i + 1, self.ant_num[0], self.simulation_time[0], self.station_num[0], self.energy_rate[0], 0, 0, None, False])
                    continue
                path, time_cost, exe_time = aco.run_aco_algorithm(optimizer_interface, source_edge, target_edge,
                                                                  self.ant_num[0], self.energy_rate[0])
                print(time_cost, path)
                all_aco_time_costs.append(time_cost)
                all_aco_exe_time_costs.append(exe_time)
                experiment_id = f"{self.ant_num[0]}-{i + 1}"
                writer.writerow([experiment_id, self.ant_num[0], self.simulation_time[0], self.station_num[0], self.energy_rate[0], time_cost, exe_time, path, 'True'])

