import unittest
from user_info import User
from optimization import Optimization


class TestOptimization(unittest.TestCase):

    def setUp(self):
        self.net_xml_path = 'graph202311211616.net.xml'
        self.source_edge = '-1'
        self.target_edge = '19'
        self.start_mode = 'walking'
        self.ant_num = 10000
        self.iteration = 1
        self.user = User(60, False, 0)

    def test_optimization_class(self):
        # Initializing the Optimization class with the paths
        db_path = 'test_new.db'
        optimizer = Optimization(self.net_xml_path, self.user, db_path)
        print("Initialize finished")
        best_path = optimizer.run_aco_algorithm(self.source_edge, self.target_edge, self.ant_num, self.iteration,
                                                self.start_mode)

    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()
