from run_Q_learning import Q_learning_agent
from user_info import User
from optimization import Optimization


# Download the Bigtable library: pip install google-cloud-bigtable
# Download the service account, JSON file from the simulation github (AdminBT.json)
# Set environment variable, GOOGLE_APPLICATION_CREDENTIALS:
# GOOGLE_APPLICATION_CREDENTIALS="/path to/AdminBT.json"


class Q_RUN:

    def __init__(self):
        self.start_edge = '361450282'
        self.destination_edge = "-110407380#1"

        self.net_xml_path = 'DCC.net.xml'
        self.start_mode = 'walking'
        self.station_num = 10
        self.energy_rate = 1
        self.simulation_time = 20000

        self.episodes = 2000
        self.iteration = 1
        self.db_path = 'test_new.db'
        self.user = User(60, True, 0, 20)

    def run_Q(self):
        optimizer = Optimization(self.net_xml_path, self.user, self.db_path, self.simulation_time, self.station_num,
                                 self.start_edge, self.destination_edge)
        graph = optimizer.new_graph

        best_route, best_modes, total_time_cost, execution_time, find = Q_learning_agent.run_q_learning(optimizer, self.start_edge, self.destination_edge, self.episodes, self.energy_rate)
        print("The total time cost is:", total_time_cost, 'seconds')
        if not find:
            print("Failed to find a valid path")
        # print(best_route)

        # else:
        #     print(best_route)
        #     print(best_modes)
        #     print(total_time_cost)
        #     print(find)


if __name__ == "__main__":
    Q_run_instance = Q_RUN()
    Q_run_instance.run_Q()
