import unittest
from user_info import User
import matplotlib.pyplot as plt
from optimization import Optimization



class Analysis(unittest.TestCase):
    def setUp(self):
        self.net_xml_path = 'DCC.net.xml'
        self.source_edge = '361450282'
        self.target_edge = "-110407380#1"
        self.start_mode = 'walking'
        self.ant_num = [100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000]
        # self.ant_num = [100, 150]
        self.iteration = 1
        self.db_path = 'test_new.db'
        self.user = User(60, True, 0, 20)

    def test_run_optimizer(self):
        optimizer_interface = Optimization(self.net_xml_path, self.user, self.db_path, self.source_edge,
                                           self.target_edge)
        graph = optimizer_interface.new_graph
        test_size = 30

        # Storing all execution times and time costs
        all_aco_exe_times = []
        all_aco_time_costs = []

        for ant_num in self.ant_num:
            aco_time_costs = []
            aco_exe_time = []
            for _ in range(test_size):
                path, time_cost, ex_time = optimizer_interface.run_aco_algorithm(self.source_edge, self.target_edge,
                                                                                 ant_num, self.iteration)
                aco_time_costs.append(time_cost)
                aco_exe_time.append(ex_time)
            all_aco_exe_times.append(aco_exe_time)
            all_aco_time_costs.append(aco_time_costs)

        # Now all_aco_exe_times and all_aco_time_costs have the raw data for all runs
        self.plot_aco_performance_2d(self.ant_num, all_aco_time_costs, all_aco_exe_times)

    def plot_aco_performance_2d(self, ant_nums, all_aco_time_costs, all_aco_exe_times):
        plt.rcParams.update({'font.size': 14})

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

        # Customize the boxplot appearance
        boxprops = dict(linestyle='-', linewidth=1.5, color='black', facecolor='cornflowerblue')
        medianprops = dict(linestyle='-', linewidth=1.5, color='darkblue')
        flierprops = dict(marker='o', color='black', alpha=0.5)

        # Boxplot for Execution Time Differences
        bp1 = ax1.boxplot(all_aco_exe_times, positions=ant_nums, widths=35, boxprops=boxprops,
                          medianprops=medianprops, flierprops=flierprops, patch_artist=True)
        ax1.set_xticklabels(ant_nums, rotation=45, ha='right')
        ax1.set_xlabel('Number of Ants', fontsize=16)
        ax1.set_ylabel('Execution Time Difference (seconds)', fontsize=16)
        ax1.set_title('ACO Performance: Execution Time Variability vs. Number of Ants', fontsize=18)
        ax1.grid(True, linestyle='--', which='major', color='grey', alpha=0.7)

        # Boxplot for Time Cost Differences
        bp2 = ax2.boxplot(all_aco_time_costs, positions=ant_nums, widths=35, boxprops=boxprops,
                          medianprops=medianprops, flierprops=flierprops, patch_artist=True)
        ax2.set_xticklabels(ant_nums, rotation=45, ha='right')
        ax2.set_xlabel('Number of Ants', fontsize=16)
        ax2.set_ylabel('Time Cost Difference (seconds)', fontsize=16)
        ax2.set_title('ACO Performance: Time Cost Variability vs. Number of Ants', fontsize=18)
        ax2.grid(True, linestyle='--', which='major', color='grey', alpha=0.7)

        plt.subplots_adjust(hspace=0.3)
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    unittest.main()
