import queryTest



def get_simulation_data(simulation):
    queryTest.query_speed_at_time(simulation)
    # filename = "query_results-" + str(self.simulation) + ".json"
    # with open(filename, 'r') as f:
    #     data = json.load(f)
    # data_dict = {entry['edge_id']: entry for entry in data}


get_simulation_data()