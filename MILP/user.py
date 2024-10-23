class User:
    def __init__(self, user_id, user_preference, source, destination, max_station_changes, original_G):
        self.user_id = user_id
        self.user_preference = user_preference
        self.source = source
        self.destination = destination
        self.max_station_changes = max_station_changes
        station_types = ['eb', 'es', 'ec', 'walk']
        self.node_stations = {i: station_types for i in original_G.nodes}
        self.node_stations[self.source] = ['walk']
        self.node_stations[self.destination] = ['walk']
