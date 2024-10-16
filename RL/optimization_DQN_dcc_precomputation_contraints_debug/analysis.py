import statistics
import unittest
import time as tm
from user_info import User
import matplotlib.pyplot as plt
from optimization import Optimization
from Q_learning_agent import MultiModalQLearningAgent


class Analysis(unittest.TestCase):
    def setUp(self):
        self.net_xml_path = 'DCC.net.xml'
        self.source_edge = '361450282'
        self.target_edge = "-110407380#1"
        self.start_mode = 'walking'
        self.ant_num = [100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250]
        self.episodes = [500, 1000, 1500, 2000]
        self.iteration = 1
        self.db_path = 'test_new.db'
        self.user = User(60, True, 0, 20)


    def test_run_optimizer(self):
        # Initializing the Optimization class with the paths
        optimizer_interface = Optimization(self.net_xml_path, self.user, self.db_path, self.source_edge,
                                           self.target_edge)
        graph = optimizer_interface.new_graph
        test_size = 30
        # Q learning average
        # Q_avg_times = []
        # Q_avg_exe_times = []
        # Q_times = []
        # Q_exe_times = []
        # for i in range(0, test_size):
        #     for episode in self.episodes:
        #         agent = MultiModalQLearningAgent(graph)
        #         start_time = tm.time()
        #         agent.learn(self.source_edge, self.target_edge, episode)
        #         end_time = tm.time()
        #         time = agent.print_optimal_route(self.source_edge, self.target_edge)
        #         Q_exe_times.append(end_time - start_time)
        #         Q_times.append(time)
        #     Q_avg_times.append(statistics.mean(Q_times))
        #     Q_avg_exe_times.append(statistics.mean(Q_exe_times))
        # self.plot_rl_performance_2d(self.episodes, Q_avg_times, Q_avg_exe_times)

        # ACO average
        aco_time_costs = []
        aco_exe_time = []
        aco_avg_times = []
        aco_avg_exe_times = []

        for ant_num in self.ant_num:
            aco_time_costs = []
            aco_exe_time = []
            for _ in range(test_size):
                path, time_cost, ex_time = optimizer_interface.run_aco_algorithm(self.source_edge, self.target_edge,
                                                                                 ant_num, self.iteration)
                aco_time_costs.append(time_cost)
                aco_exe_time.append(ex_time)
            aco_avg_times.append(statistics.mean(aco_time_costs))
            aco_avg_exe_times.append(statistics.mean(aco_exe_time))

        print("exe time \n", aco_exe_time)
        print("time costs \n", aco_time_costs)

        # self.plot_aco_performance_3d(self.ant_num, aco_time_costs, aco_exe_time)
        self.plot_aco_performance_2d(self.ant_num, aco_avg_times, aco_avg_exe_times)

    def plot_aco_performance_2d(self, ant_nums, aco_time_costs, aco_exe_time):
        plt.rcParams.update({'font.size': 14})  # Adjust font size globally for the plot

        # Creating subplots with 2 rows and 1 column
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Plotting execution time against number of ants
        ax1.plot(ant_nums, aco_exe_time, 'ro-', label='Execution Time')
        ax1.set_xlabel('Number of Ants', fontsize=16)
        ax1.set_ylabel('Execution Time (seconds)', fontsize=16)
        ax1.set_title('ACO Performance: Execution Time vs. Number of Ants', fontsize=18)
        ax1.tick_params(axis='both', which='major', labelsize=14)
        ax1.legend()
        ax1.grid(True)

        # Plotting time cost against number of ants
        ax2.plot(ant_nums, aco_time_costs, 'bs-', label='Time Cost')
        ax2.axhline(y=3084.136579657965, color='g', linestyle='--', label='Reference Value')
        ax2.set_xlabel('Number of Ants', fontsize=16)
        ax2.set_ylabel('Time Cost (seconds)', fontsize=16)
        ax2.set_title('ACO Performance: Time Cost vs. Number of Ants', fontsize=18)
        ax2.tick_params(axis='both', which='major', labelsize=14)
        ax2.legend()
        ax2.grid(True)

        # Adjust layout to prevent overlap
        plt.subplots_adjust(hspace=0.3)
        plt.tight_layout()
        plt.show()

    def plot_rl_performance_2d(self, episodes, rl_time_costs, rl_exe_time):
        plt.rcParams.update({'font.size': 14})  # Adjust font size globally for the plot

        # Creating subplots with 2 rows and 1 column
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Plotting execution time against number of ants
        ax1.plot(episodes, rl_exe_time, 'ro-', label='Execution Time')
        ax1.set_xlabel('Number of Episodes', fontsize=16)
        ax1.set_ylabel('Execution Time (seconds)', fontsize=16)
        ax1.set_title('Q-learning Performance: Execution Time vs. Number of Episodes', fontsize=18)
        ax1.tick_params(axis='both', which='major', labelsize=14)
        ax1.legend()
        ax1.grid(True)

        # Plotting time cost against number of ants
        ax2.plot(episodes, rl_time_costs, 'bs-', label='Time Cost')
        ax2.axhline(y=3084.136579657965, color='g', linestyle='--', label='Reference Value')
        ax2.set_xlabel('Number of Episodes', fontsize=16)
        ax2.set_ylabel('Time Cost (seconds)', fontsize=16)
        ax2.set_title('Q_learning Performance: Time Cost vs. Number of Episodes', fontsize=18)
        ax2.tick_params(axis='both', which='major', labelsize=14)
        ax2.legend()
        ax2.grid(True)

        # Adjust layout to prevent overlap
        plt.subplots_adjust(hspace=0.3)
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    unittest.main()
