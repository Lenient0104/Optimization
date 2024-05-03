import unittest
from optimization import Optimization


class TestOptimization(unittest.TestCase):

    def setUp(self):
        self.net_xml_path = 'graph202308142106.net.xml'
        self.edges_csv_path = 'graph202308142106_edges.csv'
        self.source_node = '5355515613'
        self.target_node = '3562601817'
        self.max_time_limit = 500

    def test_optimization_class(self):
        # Initializing the Optimization class with the paths
        optimizer = Optimization(self.net_xml_path, self.edges_csv_path)

        # Ensuring graph G is built
        self.assertIsNotNone(optimizer.G)

        # Testing constrained_shortest_path method
        path = optimizer.constrained_shortest_path(self.source_node, self.target_node, self.max_time_limit)
        print(path)
        optimizer.plot_graph_with_path(path, optimizer.G)
        self.assertIsNotNone(path)

        # Check if the path is really the shortest path with the provided constraint
        total_time = sum(optimizer.G[u][v]['weight'] for u, v in zip(path[:-1], path[1:]))
        self.assertTrue(total_time <= self.max_time_limit)

        # Testing the dijkstra_top_three_routes method
        top_three_routes = optimizer.dijkstra_top_three_routes(self.source_node, self.target_node, self.max_time_limit, optimizer.G)
        self.assertIsNotNone(top_three_routes)
        self.assertTrue(len(top_three_routes) <= 3)

        for route in top_three_routes:
            total_time = sum(optimizer.G[u][v]['weight'] for u, v in zip(route[:-1], route[1:]))
            self.assertTrue(total_time <= self.max_time_limit)

    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()
