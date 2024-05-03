import unittest
from user_info import User
import matplotlib.pyplot as plt
from optimization import Optimization


class TestOptimization(unittest.TestCase):

    def setUp(self):
        self.net_xml_path = 'DCC.net.xml'
        self.source_edge = '3789374#3'
        self.target_edge = "-361450282"
        self.start_mode = 'walking'
        self.ant_num = [100, 150, 200, 250, 300]
        self.iteration = 1
        self.user = User(60, True, 0, 20)

    def test_optimization_class(self):
        # Initializing the Optimization class with the paths
        db_path = 'test_new.db'
        optimizer = Optimization(self.net_xml_path, self.user, db_path, self.source_edge, self.target_edge)
        print("Initialize finished")
        # best_path = optimizer.run_aco_algorithm(self.source_edge, self.target_edge, self.ant_num, self.iteration,
        #                                         self.start_mode)
        # possible_paths = optimizer.find_all_routes_between(self.source_edge, self.target_edge)
        # print(possible_paths)
        best_time_costs = []
        for ant_num in self.ant_num:
            path, time_cost = optimizer.run_aco_algorithm(self.source_edge, self.target_edge, ant_num, self.iteration)
            best_time_costs.append(time_cost)

        plt.figure(figsize=(10, 6))
        plt.plot(self.ant_num, best_time_costs)
        plt.title("Time costs Over number of Ants Used")
        plt.xlabel("Number of Ants")
        plt.ylabel("Time Cost")
        plt.savefig("time_costs.png")
        plt.grid(True)
        # plt.show()


    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()
