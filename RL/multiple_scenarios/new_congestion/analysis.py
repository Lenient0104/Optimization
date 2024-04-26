import csv
import unittest
from user_info import User
import matplotlib.pyplot as plt
from optimization import Optimization


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
        # 从文件中读取OD对
        with open('od_pairs.csv', 'r') as file:
            reader = csv.reader(file)
            od_pairs = [tuple(row) for row in reader]

        test_size = len(od_pairs)  # 控制最大测试数量

        all_aco_exe_time_costs = []
        all_aco_time_costs = []

        with open('ACO_results_20congestion.csv', 'w', newline='') as file:
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
                        experiment_id = f"{ant_num}-{i + 1}"
                        writer.writerow([experiment_id, ant_num, time_cost, exe_time])


        self.plot_aco_performance_2d(self.ant_num, all_aco_time_costs)

    def plot_aco_performance_2d(self, ant_nums, all_aco_time_costs):
        plt.rcParams.update({'font.size': 14})
        plt.figure(figsize=(10, 6))
        boxprops = dict(linestyle='-', linewidth=1.5, color='black', facecolor='cornflowerblue')
        medianprops = dict(linestyle='-', linewidth=1.5, color='darkblue')
        flierprops = dict(marker='o', color='black', alpha=0.5)

        # Boxplot for Time Cost Differences
        bp = plt.boxplot(all_aco_time_costs, positions=ant_nums, widths=35, boxprops=boxprops,
                         medianprops=medianprops, flierprops=flierprops, patch_artist=True)
        plt.xticks(ant_nums, rotation=45, ha='right')
        plt.xlabel('OD Pairs', fontsize=16)
        plt.ylabel('Travel Time Cost (seconds)', fontsize=16)
        plt.title('ACO Performance: Travel Time Cost vs. Different OD pairs', fontsize=18)
        plt.grid(True, linestyle='--', which='major', color='grey', alpha=0.7)

        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    unittest.main()
