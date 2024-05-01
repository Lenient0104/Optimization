import csv
import unittest
from user_info import User
import matplotlib.pyplot as plt
from optimization_new import Optimization


class Analysis(unittest.TestCase):
    def setUp(self):
        self.net_xml_path = 'DCC.net.xml'
        self.start_mode = 'walking'
        self.ant_num = [350]
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

        with open('results/ACO_results_40speednew.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Experiment ID', 'Number of Ants', 'Travel Time Cost (seconds)', 'Execution Time (seconds)'])

            for ant_num in self.ant_num:
                aco_time_costs = []
                aco_exe_costs = []
                for i in range(test_size):
                    print(i)
                    source_edge, target_edge = od_pairs[i]
                    optimizer_interface = Optimization(self.net_xml_path, self.user, self.db_path, source_edge, target_edge)
                    path, time_cost, exe_time = optimizer_interface.run_aco_algorithm(source_edge, target_edge, ant_num)
                    print(time_cost, path)
                    if exe_time != 0 and time_cost != 'inf':
                        all_aco_time_costs.append(time_cost)
                        all_aco_exe_time_costs.append(exe_time)
                        experiment_id = f"{i + 1}"
                        writer.writerow([experiment_id, ant_num, time_cost, exe_time])




if __name__ == '__main__':
    unittest.main()
