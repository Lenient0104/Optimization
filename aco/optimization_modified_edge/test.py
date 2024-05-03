import unittest
from optimization import Optimization


class TestOptimization(unittest.TestCase):

    def setUp(self):
        self.net_xml_path = 'test.net.xml'
        self.edges_csv_path = 'test.csv'
        self.source_edge = '129'
        self.target_edge = '-46'
        self.max_time_limit = 9999

    def test_optimization_class(self):
        # Initializing the Optimization class with the paths
        optimizer = Optimization(self.net_xml_path, self.edges_csv_path)

        # Ensuring graph G is built
        self.assertIsNotNone(optimizer.G)

        # Testing constrained_shortest_path method
        path = optimizer.constrained_shortest_path(self.source_edge, self.target_edge, self.max_time_limit)
        print(path)
        # edge_path = optimizer.convert_node_path_to_edge_path(path)
        # print(edge_path)
        # self.assertIsNotNone(path)

        # Check if the path is really the shortest path with the provided constraint
        total_time = sum(optimizer.G[u][v]['weight'] for u, v in zip(path[:-1], path[1:]))
        self.assertTrue(total_time <= self.max_time_limit)

        # Testing the dijkstra_top_three_routes method
        # Get 'to' node of the source edge and 'from' node of the target edge
        # source_node = optimizer.edges[self.source_edge]['from']
        # target_node = optimizer.edges[self.target_edge]['from']

        # Testing the dijkstra_top_three_routes method
        # top_three_routes = optimizer.dijkstra_top_three_routes(source_node, target_node, self.max_time_limit, optimizer.G)
        # self.assertIsNotNone(top_three_routes)
        # self.assertTrue(len(top_three_routes) <= 3)

        # for route in top_three_routes:
        #     total_time = sum(optimizer.G[u][v]['weight'] for u, v in zip(route[:-1], route[1:]))
        #     self.assertTrue(total_time <= self.max_time_limit)

    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()