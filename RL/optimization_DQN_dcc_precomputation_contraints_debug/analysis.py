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
        self.ant_num = [100, 125, 150, 175, 200, 250, 300]
        self.episodes = [500, 1000, 1500, 2000]
        self.iteration = 1
        self.db_path = 'test_new.db'
        self.user = User(60, True, 0, 20)


    def test_run_optimizer(self):
        # Initializing the Optimization class with the paths
        optimizer_interface = Optimization(self.net_xml_path, self.user, self.db_path, self.source_edge,
                                           self.target_edge)
        graph = optimizer_interface.new_graph

        Q_times = []
        Q_exe_times = []
        for episode in self.episodes:
            agent = MultiModalQLearningAgent(graph)
            start_time = tm.time()
            agent.learn(self.source_edge, self.target_edge, episode)
            end_time = tm.time()
            time = agent.print_optimal_route(self.source_edge, self.target_edge)
            Q_exe_times.append(end_time - start_time)
            Q_times.append(time)

        self.plot_rl_performance_2d(self.episodes, Q_times, Q_exe_times)
        aco_time_costs = []
        aco_exe_time = []
        for ant_num in self.ant_num:
            path, time_cost, ex_time = optimizer_interface.run_aco_algorithm(self.source_edge, self.target_edge, ant_num,
                                                                             self.iteration)
            aco_time_costs.append(time_cost)
            aco_exe_time.append(ex_time)

        print("exe time \n", aco_exe_time)
        print("time costs \n", aco_time_costs)

        # self.plot_aco_performance_3d(self.ant_num, aco_time_costs, aco_exe_time)
        self.plot_aco_performance_2d(self.ant_num, aco_time_costs, aco_exe_time)

    def plot_aco_performance_3d(self, ant_nums, aco_time_costs, aco_exe_time):
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Set labels for axes
        ax.set_xlabel('Number of Ants')
        ax.set_ylabel('Execution Time (seconds)')
        ax.set_zlabel('Time Cost (seconds)')

        # Ensure ant_nums is a list of integers if it's not already
        ant_nums_int = [int(ant) for ant in ant_nums]

        # Plotting the data points
        ax.scatter(ant_nums_int, aco_exe_time, aco_time_costs, color='red', s=100, label='ACO Performance')

        # Connect the main nodes with a line
        ax.plot(ant_nums_int, aco_exe_time, aco_time_costs, color='black', linewidth=2)

        # Then set the x-ticks to the ant_nums list
        ax.set_xticks(ant_nums_int)
        ax.set_xticklabels(ant_nums_int)

        # Drawing lines to the axes and marking the intersection points
        for x, y, z in zip(ant_nums_int, aco_exe_time, aco_time_costs):
            ax.plot([x, x], [y, y], [0, z], 'gray', linestyle='--', alpha=0.5)  # Line to z-axis
            ax.plot([0, x], [y, y], [z, z], 'gray', linestyle='--', alpha=0.5)  # Line to x-axis
            ax.scatter([0], [y], [z], color='blue', s=15)  # Mark on y-axis plane
            ax.scatter([x], [y], [0], color='blue', s=15)  # Mark on z-axis plane

        # Set the view angle for better visualization
        # ax.view_init(elev=20, azim=-35)

        print(ax.get_xlim(), ax.get_ylim(), ax)

        plt.title("ACO Performance: Execution Time and Time Cost vs. Number of Ants")
        plt.legend()
        plt.show()

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

    def visual_comparison(self, aco_time_costs, RL_time_cost):
        time_differences = []
        for aco_cost in aco_time_costs:
            time_differences.append(aco_cost - RL_time_cost)

        plt.figure(figsize=(10, 6))
        plt.plot(self.ant_num, time_differences)
        plt.title("Optimality of Objective Value difference between RL optimizer and ACO optimizer over number of "
                  "Ants Used")
        plt.xlabel("Number of Ants")
        plt.ylabel("Time Cost Difference (seconds)")
        plt.savefig("comparison")
        # plt.show()


if __name__ == '__main__':
    unittest.main()
