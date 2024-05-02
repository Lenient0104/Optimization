import csv
import unittest
from user_info import User
import matplotlib.pyplot as plt
from optimization_new import Optimization

# Download the Bigtable library: pip install google-cloud-bigtable
# Download the service account, JSON file from the simulation github (AdminBT.json)
# Set environment variable, GOOGLE_APPLICATION_CREDENTIALS:
# GOOGLE_APPLICATION_CREDENTIALS="/path to/AdminBT.json"

class Analysis(unittest.TestCase):
    def setUp(self):
        self.net_xml_path = 'DCC.net.xml'
        self.start_mode = 'walking'
        self.ant_num = [350]
        self.simulation_time = [40000, 50000, 60000, 70000, 80000]
        self.episodes = [500, 1000, 1500, 2000]
        self.iteration = 1
        self.db_path = 'test_new.db'
        self.user = User(60, True, 0, 20)

    def test_run_optimizer(self):
        with open('od_pairs_500.csv', 'r') as file:
            reader = csv.reader(file)
            od_pairs = [tuple(row) for row in reader]

        test_size = len(od_pairs)
        all_aco_exe_time_costs = []
        all_aco_time_costs = []

        with open('results/ACO-results-simulation_time_0_new.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Experiment ID', 'Number of Ants', 'Simulation Time', 'Travel Time Cost (seconds)', 'Execution Time (seconds)', 'Find'])

            for simulation in self.simulation_time:
                aco_time_costs = []
                aco_exe_costs = []
                for i in range(test_size):
                    print(simulation, i)
                    source_edge, target_edge = od_pairs[i]
                    optimizer_interface = Optimization(self.net_xml_path, self.user, self.db_path, simulation, source_edge, target_edge)
                    graph = optimizer_interface.new_graph
                    if graph is None:
                        writer.writerow([i + 1, self.ant_num[0], simulation, 0, 0, False])
                        continue
                    path, time_cost, exe_time = optimizer_interface.run_aco_algorithm(source_edge, target_edge, self.ant_num[0])
                    print(time_cost, path)
                    all_aco_time_costs.append(time_cost)
                    all_aco_exe_time_costs.append(exe_time)
                    experiment_id = f"{self.ant_num[0]}-{i + 1}"
                    writer.writerow([experiment_id, self.ant_num[0], simulation, time_cost, exe_time, 'True'])




if __name__ == '__main__':
    unittest.main()
