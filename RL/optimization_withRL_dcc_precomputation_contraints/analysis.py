import unittest
from user_info import User
import matplotlib.pyplot as plt
from optimization import Optimization
from Q_learning_agent import MultiModalQLearningAgent


class Analysis(unittest.TestCase):
    def setUp(self):
        self.net_xml_path = 'DCC.net.xml'
        self.source_edge = '3789374#3'
        self.target_edge = "-361450282"
        self.start_mode = 'walking'
        self.ant_num = [100, 150, 200, 250, 300]
        self.iteration = 1
        self.db_path = 'test_new.db'
        self.user = User(60, True, 0, 20)

    def test_run_optimizer(self):
        # Initializing the Optimization class with the paths
        optimizer_interface = Optimization(self.net_xml_path, self.user, self.db_path, self.source_edge,
                                           self.target_edge)
        graph = optimizer_interface.new_graph

        aco_time_costs = []
        for ant_num in self.ant_num:
            path, time_cost = optimizer_interface.run_aco_algorithm(self.source_edge, self.target_edge, ant_num,
                                                                    self.iteration)
            aco_time_costs.append(time_cost)

        RL_agent = MultiModalQLearningAgent(graph)
        RL_agent.learn(self.source_edge, self.target_edge)
        RL_time_cost = RL_agent.print_optimal_route(self.source_edge, self.target_edge)
        self.visual_comparison(aco_time_costs, RL_time_cost)

    def visual_comparison(self, aco_time_costs, RL_time_cost):
        time_differences = []
        for aco_cost in aco_time_costs:
            time_differences.append(aco_cost - RL_time_cost)

        plt.figure(figsize=(10, 6))
        plt.plot(self.ant_num, time_differences)
        plt.title("Time costs difference between RL optimizer and ACO optimizer over number of Ants Used")
        plt.xlabel("Number of Ants")
        plt.ylabel("Time Cost Difference (seconds)")
        plt.savefig("comparison")
        # plt.show()


if __name__ == '__main__':
    unittest.main()
