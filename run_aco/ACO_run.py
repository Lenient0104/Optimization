from run_aco import aco
from user_info import User
from optimization import Optimization

# Download the Bigtable library: pip install google-cloud-bigtable
# Download the service account, JSON file from the simulation github (AdminBT.json)
# Set environment variable, GOOGLE_APPLICATION_CREDENTIALS:
# GOOGLE_APPLICATION_CREDENTIALS="/path to/AdminBT.json"


class ACO_RUN:

    def __init__(self):
        self.start_edge = '361450282'
        self.destination_edge = "-110407380#1"

        self.net_xml_path = 'DCC.net.xml'
        self.start_mode = 'walking'
        self.ant_num = 2000
        self.station_num = 10
        self.energy_rate = 1
        self.simulation_time = 20000

        self.episodes = 1000
        self.iteration = 1
        self.db_path = 'test_new.db'
        self.user = User(60, True, 0, 20)

    def run_aco(self):
        optimizer_interface = Optimization(self.net_xml_path, self.user, self.db_path, self.simulation_time, self.station_num, self.start_edge, self.destination_edge)
        graph = optimizer_interface.new_graph
        if graph is None:
            print("The graph can not be constructed")
        else:
            path, time_cost, exe_time = aco.run_aco_algorithm(optimizer_interface, self.start_edge, self.destination_edge,
                                                              self.ant_num, self.energy_rate)
            print("The total time cost is:", time_cost, "seconds")
            print("The optimal path is:", path)


if __name__ == "__main__":
    aco_run_instance = ACO_RUN()
    aco_run_instance.run_aco()