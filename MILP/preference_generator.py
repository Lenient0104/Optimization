class PreferenceGenerator:
    def __init__(self, G, station_types):
        self.G = G  # original graph
        self.station_types = station_types  # station_types = ['eb', 'es', 'ec', 'walk']
        self.node_station_pair = {}

    def generate_node_preferences(self):
        preferred_station = {}
        preferred_nodes = ['361409608#3', '3791905#2', '-11685016#2', '369154722#2', '244844370#0', '37721356#0',
                           '74233405#1', '129774671#0', '23395388#5', '-64270141', '18927706#0', '-42471880',
                           '67138626#1', '41502636#0', '-75450412', '-23347664#1', '14151839#3', '-12341242#1',
                           '-13904652',
                           '-47638297#3']

        for i in self.G.nodes:
            if i in preferred_nodes:
                # num_preferred = random.randint(1, len(self.station_types) - 1)
                # preferred_types = [st for st in self.station_types if st != 'walk']
                # 为每个站点生成唯一的 station_types 列表
                # num_preferred = random.randint(1, len(self.station_types))  # 随机生成 1 到全部 station_types
                # preferred_types = random.sample(self.station_types, num_preferred)
                preferred_types = ['ec', 'es', 'eb', 'walk']
                preferred_station[i] = preferred_types  # random.sample(preferred_types, num_preferred)
                if 'walk' not in preferred_station[i]:
                    preferred_station[i].append('walk')
                self.node_station_pair[i] = preferred_station[i]
            else:
                preferred_station[i] = ['']

        return preferred_station, preferred_nodes
