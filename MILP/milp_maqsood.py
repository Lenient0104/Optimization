import random
import pulp
import time
import numpy as np  # 1.21.0
import random
# import matplotlib.pyplot as plt  # 3.3.2
from tqdm import tqdm
import xml.etree.ElementTree as ET
import networkx as nx
import re
import json
import csv

class GraphHandler:
    def __init__(self, file_path):
        self.file_path = file_path
        self.G = self.create_graph_from_net_xml()

    def create_graph_from_net_xml(self):
        unique_edges = []
        connections = []
        pattern = r"^[A-Za-z]+"
        tree = ET.parse(self.file_path)
        root = tree.getroot()

        edge_detail_map = {}
        for edge in tqdm(root.findall('edge'), desc="Processing edges"):
            for lane in edge.findall('lane'):
                edge_id = edge.attrib['id']
                edge_detail_map[edge_id] = {
                    'length': float(lane.attrib['length']),
                    'speed_limit': float(lane.attrib['speed']),
                    'shape': lane.attrib['shape'],
                }

        for conn in tqdm(root.findall('connection'), desc="Processing connections", mininterval=1.0):
            pairs = []
            from_edge = conn.get('from')
            to_edge = conn.get('to')

            if from_edge.startswith(":") or re.match(pattern, from_edge) is not None:
                continue
            if from_edge not in unique_edges and from_edge != 'gneE29':
                unique_edges.append(from_edge)
            if to_edge not in unique_edges:
                unique_edges.append(to_edge)

            pairs.append(from_edge)
            pairs.append(to_edge)
            connections.append(pairs)

        G = nx.DiGraph()
        for edge in unique_edges:
            G.add_node(edge, **edge_detail_map.get(edge, {}))

        for from_edge, to_edge in connections:
            if from_edge in edge_detail_map and to_edge in edge_detail_map:
                length = edge_detail_map[from_edge]['length']
                G.add_edge(from_edge, to_edge, weight=length)

        return G

    def get_graph(self):
        return self.G

class PreferenceGenerator:
    def __init__(self, G, station_types, num_preferred_nodes):
        self.G = G
        self.station_types = station_types
        self.num_preferred_nodes = num_preferred_nodes

    def generate_node_preferences(self):
        preferred_station = {}
        preferred_nodes = ['361409608#3', '3791905#2', '-11685016#2', '369154722#2', '244844370#0', '37721356#0',
                          '74233405#1', '129774671#0', '23395388#5', '-64270141'] #random.sample(list(self.G.nodes), self.num_preferred_nodes)
        #20
        # preferred_nodes = ['361409608#3', '3791905#2', '-11685016#2', '369154722#2', '244844370#0', '37721356#0',
        #                       '74233405#1', '129774671#0', '23395388#5', '-64270141', '18927706#0', '-42471880',
        #                       '67138626#1', '41502636#0', '-75450412', '-23347664#1', '14151839#3', '-12341242#1', '-13904652',
        #                       '-47638297#3']
        #50
        # preferred_nodes = ['-52881219#6', '3788262#0', '-36819763#3', '151482073', '-44432110#1', '-45825602#2', '-37721078#5', '-4834964#1', '59433202', '-441393227#1', 
        #                    '20485040#0', '29333277', '-38062537', '37812705#0', '3708641#0', '4934504#3', '-8232242#0', '254372188#2',
        #                      '-114208220#0', '-370381577#2', '228694841', '-22716429#2', '129068838', '-191042862#1', '42636056', '7849019#2', '556290239#1', '-63844803#0', '303325200#2', '4395609#0', '-575044518', '77499019#1', '-38375201#6', '129211829#0', '-228700735', '23086918#0', '-4825259#1', '-481224216#4', '-5825856#7', '-7989926#3', '4385842#0', '30999830#4', '529166764#0', '334138114#8', '-326772508#1', '3708865#8', '25787652', '55668420#1', '55814730#3', '-561862256']
        #100
        # preferred_nodes = ['128521563#0', '401904747', '31897169#3', '3284827#0', '-52241896', '372838330#2', '-376465145#1', '432565295#5', '24644260#0', '4540245#3', '4934643#0', '25536089#1', '-67139161#4', '30988145#0', '-282195032#5', '-48205720#2', '-128450676#3', '-115656107', '75741007', '32346509', '3818299#3', '-128946815#7', '-111746918', '38864100#0', '31350775#4', '662531575#0', '-43870791#1', '44385285', '-44436781#1', 
        #                    '129211827#0', '94838278#4', '-43175828#0', '37747500#5', '-129211821#4', '-25761629', '125813654#1', '369359214', '369154720#2', '30583814#1', '-365945819#0', '4541293#0', '520012239#0', '81982578#10', '-7919284#10', '7919276#0', '-326874928#1', '147077493#0', '45619558', '-127220738', '-14047195', '-45530928#0', '-372245390#1', '23501248#4', '14047190#0', '-26150087#3', '-7977648#1', '55668414#1', '256024070#3', '-129211821#7', 
        #                    '3191572#5', '-369977727#1', '677070805#4', '-3191572#1', '553868718#1', '-48226178#0', '-369622780', '-20848460#1', '174092689#0', '228694841', '64270037#0', '4385874#1', '-238740928#2', '45530928#1', '-80478293#2', '432565295#3', '-317027066#0', '-3818568#3', '-39247679#3', '285487940', '-4540248#0', '-3708538#2', '-129211838', '231340188#0', '-13998367', '-3789251#3', '6653000#5', '317219094#1', '48454248#0', '13878225#0', '-495995815',
        #                      '-94838278#4', '-38090686#2', '32201932', '-30454721#1', '-147463637#1', '-49728145#0', '143866317#0', '-14039614', '-636824321', '-228665722']
        # 200
        # preferred_nodes =['-14047498#3', '128676689#5', 'gneE25', '-3788879#2', '3266498', '-7919280#1', '111286353#2', '4834963#1', '3791210#1', '7919275#3', '-4825239#1', '497486086', '-44409757#0', '39540888#1', '283633789', '4395858#1', '-37721078#6', '14047759#0', '8045273#0', '-45406018', '45406017', '-222609066#1', '-32016067#0', '-44705513', '-13866422', '-300664821#4', '-394233223#2', '29561733#0', '-362985327#2', '-41592110', '317554424#4', '-319556828#0', '41592114', '-4215597#5', '230478913', '4396062#0', '137621708#0', '-4825332', '317101094#12', '372256216#3', '-28002618#0', '4934662#8', '71423403#2', '3708534#0', '-30999712', '-49728136#1', '-20848457', '30655095#4', '231612950#0', '-3708465#1', '37747500#3', '14039609#0', '14047769#0', '-44404539#1', '-3191573#4', '41851537#1', '37746962', '40047164#0', '-7919286#1', '16247623', '37722081#0', '32631306#0', '55668395#0', '121249650#6', '362939722', '3708878#1', '-38574544#2', '3763679#1', '-4934662#7', '114207514#0', '52241890#1', '13904682#3', '41502636#0', '-114277935#1', '-51128241', '-114208222#0', '-44432110#0', '3788877#0', '-370154355', '-49728142#2', '-4691594', '-115656107', '30389516#0', '-3191572#5', '4395933#6', '-46895350#2', '13904652', '128450677#2', '-405364257#0', '129211857#3', '-308731562#1', '-4475686#2', '-4264251', '375581292#0', '362939725', '530413782', '-40701185', '23011304#1', '-23076566', '4287188#2', '-310303150', '-3266486#10', '-3708865#5', '4937390', '6273610#2', '-317697975#0', '-8384431#7', '-38122989', '-405546346', '-145344471#0', '-31008788', '3708468', '-22566801#0', '13871411', '-23011304#1', '-3936651', '4934682#0', '529166766#0', '379071737#0', '-282195032#5', '43316112#0', '-7977023#4', '-48985672', '-55814729#3', '13904645#0', '-30047545#3', '13878308#0', '-37721078#4', '31992229#1', '4919615#4', '-4571527#3', '3791745#4', '20848454#2', '-30999830#0', '37692199#0', '18928030#0', '110407380#0', '481221025', '289876870', '32016067#1', '-140673640', '-317145973#0', '41772640', '-334138117#0', '-38864096#1', '368505007#2', '23395388#5', '125813655#0', '-10892467#0', '-4934446#0', '77499019#1', '-227677863#2', '-31897170', '317101094#7', '25761610#0', '25535969#2', '4401701#1', '7421646#8', '125863788', '-185127488#2', '29561733#3', '-24234804', '4395994#0', '-348538605#0', '-67139161#4', '20848425#0', '-7989949#1', '-375581292#0', '-3708538#1', '23567535#2', '-48520199', '2689742', '-18928030#2', '430591268#3', '-7919276#4', '25731437', '24776090#1', '-80439756#2', '317219091#3', '-3708543', '143866317#0', '-228731431#0', '481221028#4', '-3788877#4', '-376394516#0', '423283956#0', '-32117786#4', '44069390#3', '-3285420#2', '38574621#1', '4571527#0', '-23567536#1', '37692490#3', '4899745#0', '13871250', '129211835#0', '282195032#6', '-38122987#2', '-10575480#2', '-227767347#1']
        # 300
        # preferred_nodes =['-528915436#0', '-683418232#1', '-13904703', '368505008#3', '4825258#0', '129211847#0', '40047164#0', '4834969#0', '-368505008#3', '-31992438#0', '-77499018', '-61717563#0', '233718298#1', '8232242#7', '282243997', '-20848429', '-394233223#2', '33238658', '-20686243', '-3785444#0', '-22978740#8', '191042860#0', '-3788879#5', '7919278#0', '658163878#0', '302527226#0', '44069392', '238740923#0', '-31008788', '45406018', '-231620391#1', '-44402155', '32491867#1', '-4934446#0', '125880830#9', '8468553#4', '-4395609#3', '-5825858#4', '-334138114#6', '-3788308#3', '38668910#2', '-366887650#2', '-31982174#1', '-129211812#7', '-32642508#4', '-31350775#5', '-231612455#3', '37692199#0', '14047764#0', '-132759854#2', '334138117#1', '222609066#2', '135583247', '-37692198#1', '59433202', '-14047194#0', '-547766101#0', '4545229#0', '-67139161#4', '4919460', '-110545289#2', '52375061', '-7919286#1', '12341232#0', '30047536#0', '-3785494#2', '348538605#1', '52241892', 'gneE18', '38669011#0', '30655092#0', '100654686', '-30655096', '-191042862#1', '372838330#0', '129211853', '-54451008#3', '228729058', '368646813#11', '369154717#1', '317219088', '-37747137#1', '60521432', '386571734#1', '-156375816#2', '191042863', '11685019#0', '-228557885#0', '10575471#2', '-3791210#1', '-238740928#2', '-31982160#2', '32491867#2', '-41592113#0', '-300664821#4', '4926030', '-22566802#0', '80479201#0', '38574544#2', '7421646#5', '-4934444#2', '362985327#0', '-228731434', '3789571#1', '-30454725#3', '-43321718#2', '129217471#0', '49961187#0', '371391166#0', '13904648#0', '4395692', '-532427444#3', '64272633#1', '-4396062#6', '-29333723#3', '379666708', '44007601#0', '30047545#2', '3789251#0', '23567534#7', '41772640', '48455143', '401912432#2', '230751238', '-317697974', '-375758262#1', '227711231#0', '-39523885#9', '4571527#4', '14047962#0', '-20485040#4', '23920340', '30047549#2', '-30046890', '-32015727', '-30454721#5', '-51128240#5', '52375066#1', '48455139#0', '-47861482#1', '64270035#0', '111746918', '40260272#0', '13904649#1', '-8384431#1', '-5825858#1', '-63919918', '4395994#0', '30999830#0', '2110698#0', '13904645#0', '3785449#1', '-32631772#2', '-4078065#4', '283771398#1', '30999836#1', '-54451007#3', '49728145#2', '48205721#2', '29554114#0', '-4287395#0', '143866317#0', '94838285#1', '162745568#0', '30655095#5', '-4934504#3', '-260607183#1', '-48455138', '31350775#5', '4541293#0', '-64344154', '-228818578', '64270141', '-13894939#5', '37747008#0', '-8384431#3', '-4829724#2', '-285991276', '-360342135#0', '-50503783#1', '54451008#3', '24601280', '8468553#3', '-375581291#1', '-100654687', '-7977647#1', '129211832', '63919918', '-10575471#0', '-7415911#3', '55814737', '22716429#1', '14039611#1', '4571776#7', '58403835#0', '31982229#0', '37692391#0', '-23395388#6', '56681848#2', '38668910#0', '49728141', '4628794#0', '-49728136#2', '-30988144#1', '-30818795#3', '-370154352#0', '229143734', '32117788#9', '3791207#0', '4215597#0', '673136293', '4215601#0', '-317697975#2', '3789251#4', '-423711652#1', '129211824', '22978622#0', '-24776091', '-4934446#2', '-130277328#0', '-317145978', '-3708548#1', '-39117802#1', '-317101094#11', '520012241#0', '-116689248#1', '27219884#2', '14047501#0', '80439756#0', '-372256216#4', '-593087259#1', '52881850#0', '3266486#2', '-4541293#8', '80479201#2', '29561733#3', '23307145#0', '44409754', '-29266714#0', '4287295#0', '372245388#0', '3788281#0', '306917251#1', '-476166088', '-41362435', '-30999836#0', '-3708680#2', '317544589#2', '23050596', '-13871230#1', '-137621706#0', '372460213#3', '-3708471#2', '-13904650#6', '94838282', '47861482#0', '-48996164', '581774894#1', '-4926120#3', '423711652#1', '4078079#1', '94838279', '-30454724', '-30046887#0', '129211813#0', '39117800#1', '4287395#2', '83806426#4', '-37721078#5', '-3285422#0', '302617608#1', '14327903#3', '30999830#4', '33854097#2', '129211797#2', '-31897169#3', '13998368#0', '-231612950#2', '30454721#0', '-44069390#2', '-228557873', '37691265#0', '-4934627', '-279585244#0', '13995755#0', '24450827', '4263428#2', '-38062538', '-481221027#2', '50502873', '-23076537#2', '-37721078#4', '4919615#0', '348538605#0', '32631306#0', '-3708547', '-26150087#2', '-331870104#4', '-231630881#1', '-191042861']
        
        for i in self.G.nodes:
            if i in preferred_nodes:
                #num_preferred = random.randint(1, len(self.station_types) - 1)
                preferred_types = [st for st in self.station_types if st != 'walk']
                preferred_station[i] = preferred_types #random.sample(preferred_types, num_preferred)
                if 'walk' not in preferred_station[i]:
                    preferred_station[i].append('walk')
            else:
                preferred_station[i] = ['']

        return preferred_station, preferred_nodes

class OptimizationProblem:
    def __init__(self, G, node_stations, preferred_station, M, speed_dict, user_preference, congestion=1):
        self.G = G
        self.node_stations = node_stations
        self.preferred_station = preferred_station
        self.M = M
        self.speed_dict = speed_dict
        self.user_preference = user_preference
        self.congestion = congestion
        self.paths = None
        self.station_changes = None
        self.costs = None
        self.station_change_costs = None
        self.prob = None

    def setup_decision_variables(self):
        self.paths = {(i, j, s): pulp.LpVariable(f"path_{i}_{j}_{s}", 0, 1, pulp.LpBinary)
                      for i, j in self.G.edges()
                      for s in set(self.node_stations[i]).intersection(self.node_stations[j])}

        self.station_changes = {(i, s1, s2): pulp.LpVariable(f"station_change_{i}_{s1}_{s2}", 0, 1, pulp.LpBinary)
                                for i in self.G.nodes
                                for s1 in self.preferred_station[i]
                                for s2 in self.preferred_station[i] if s1 != s2}

    def setup_costs(self):
        self.costs = {}
        for i, j in self.G.edges():
            edge_weight = self.G[i][j]['weight']
            edge_id = i
            speeds = self.speed_dict[edge_id]
            
            for s in set(self.node_stations[i]).intersection(self.node_stations[j]):
                if s != 'walk':
                    # if  s not in self.user_preference:
                    if s not in self.user_preference:
                        self.costs[i, j, s] = edge_weight * 1e7
                    else:
                        if s == 'eb' or s == 'es':
                            speed = speeds['bike_speed']
                        elif s == 'ec':
                            speed = speeds['car_speed']

                        if speed == 0:
                            self.costs[i, j, s] = 1e7#float('inf')
                        else:
                            self.costs[i, j, s] = edge_weight / (speed)
                else:
                    speed = speeds['pedestrian_speed']
                    if speed == 0:
                        self.costs[i, j, s] = 1e7 #float('inf')
                    else:
                        self.costs[i, j, s] = edge_weight / speed

        self.station_change_costs = {}
        for i in self.G.nodes:
            if len(self.preferred_station[i]) > 1:
                for s1 in self.preferred_station[i]:
                    for s2 in self.preferred_station[i]:
                        if s1 != s2:
                            self.station_change_costs[i, s1, s2] = 0.1
                        else:
                            self.station_change_costs[i, s1, s2] = self.M

    def setup_problem(self, start_node, start_station, end_node, end_station, max_station_changes):
        self.prob = pulp.LpProblem("Minimize_Traversal_Cost", pulp.LpMinimize)
        self.prob += pulp.lpSum([self.paths[i, j, s] * self.costs[i, j, s] for i, j, s in self.paths]) + \
                     pulp.lpSum([self.station_changes[i, s1, s2] * self.station_change_costs[i, s1, s2] for i, s1, s2 in self.station_changes])

        for i in self.G.nodes:
            for s in station_types:
                if s in self.node_stations[i]:
                    incoming_flow = pulp.lpSum([self.paths[j, i, s] for j in self.G.predecessors(i) if (j, i, s) in self.paths])
                    outgoing_flow = pulp.lpSum([self.paths[i, j, s] for j in self.G.successors(i) if (i, j, s) in self.paths])

                    incoming_station_changes = pulp.lpSum([self.station_changes[i, s2, s] for s2 in self.node_stations[i] if (i, s2, s) in self.station_changes])
                    outgoing_station_changes = pulp.lpSum([self.station_changes[i, s, s2] for s2 in self.node_stations[i] if (i, s, s2) in self.station_changes])

                    incoming_flow += incoming_station_changes
                    outgoing_flow += outgoing_station_changes
                    
                    if i == start_node and s == start_station:
                        self.prob += outgoing_flow == 1
                        self.prob += incoming_flow == 0
                    elif i == end_node and s == end_station:
                        self.prob += incoming_flow == 1
                        self.prob += outgoing_flow == 0
                    else:
                        self.prob += incoming_flow == outgoing_flow

        self.prob += pulp.lpSum(self.station_changes.values()) <= max_station_changes

    def solve(self):
        start_time = time.time()
        # self.prob.solve(pulp.GUROBI_CMD(msg=False))
        self.prob.solve(pulp.PULP_CBC_CMD(msg=False))
        end_time = time.time()
        return self.prob, end_time - start_time

class PathFinder:
    def __init__(self, paths, station_changes, costs, station_change_costs):
        self.paths = paths
        self.station_changes = station_changes
        self.costs = costs
        self.station_change_costs = station_change_costs

    def generate_path_sequence(self, start_node, start_station, end_node, end_station):
        current_node, current_mode = start_node, start_station
        path_sequence = []
        station_change_count = 0
        destination_reached = False

        while not destination_reached:
            next_step_found = False
            for i, j, s in self.paths:
                if i == current_node and s == current_mode and pulp.value(self.paths[i, j, s]) == 1:
                    path_cost = self.costs[i, j, s]
                    path_sequence.append((i, j, s, path_cost))
                    current_node = j
                    next_step_found = True
                    break

            for i, s1, s2 in self.station_changes:
                if i == current_node and s1 == current_mode and pulp.value(self.station_changes[i, s1, s2]) == 1:
                    mode_change_cost = self.station_change_costs[i, s1, s2]
                    path_sequence.append((i, s1, s2, mode_change_cost))
                    current_mode = s2
                    station_change_count += 1
                    next_step_found = True

            if current_node == end_node and current_mode == end_station:
                destination_reached = True
            elif not next_step_found:
                print("Destination not reached. Path may be incomplete.")
                break

        return path_sequence, station_change_count

class ShortestPathComputer:
    def __init__(self, graph):
        self.graph = graph

    def compute_shortest_paths_start(self, start_node, preference_stations):
        shortest_routes_start = {}
        for station in preference_stations:
            try:
                shortest_path = nx.shortest_path(self.graph, source=start_node, target=station, weight='weight')
                shortest_routes_start[station] = (shortest_path, nx.shortest_path_length(self.graph, source=start_node, target=station, weight='weight'))
            except nx.NetworkXNoPath:
                pass
        return shortest_routes_start
    
    def compute_shortest_paths_pairs(self, preference_stations):
        all_shortest_routes_pairs = {}
        for station1 in preference_stations:
            for station2 in preference_stations:
                if station1 != station2:
                    try:
                        shortest_path = nx.shortest_path(self.graph, source=station1, target=station2, weight='weight')
                        all_shortest_routes_pairs[(station1, station2)] = (shortest_path, nx.shortest_path_length(self.graph, source=station1, target=station2, weight='weight'))
                    except nx.NetworkXNoPath:
                        pass
        return all_shortest_routes_pairs

    def compute_shortest_paths_dest(self, dest_node, preference_stations):
        shortest_routes_dest = {}
        for station in preference_stations:
            try:
                shortest_path = nx.shortest_path(self.graph, source=station, target=dest_node, weight='weight')
                shortest_routes_dest[station] = (shortest_path, nx.shortest_path_length(self.graph, source=station, target=dest_node, weight='weight'))
            except nx.NetworkXNoPath:
                pass
        return shortest_routes_dest
    
    def compute_shortest_path_start_end(self, start_node, dest_node):
        try:
            shortest_path = nx.shortest_path(self.graph, source=start_node, target=dest_node, weight='weight')
            shortest_route_start_end = (shortest_path, nx.shortest_path_length(self.graph, source=start_node, target=dest_node, weight='weight'))
            return shortest_route_start_end
        except nx.NetworkXNoPath:
            return None
    

class ReducedGraphCreator:
    def __init__(self, graph, start_node, dest_node, preference_stations, shortest_routes_start, shortest_routes_dest, all_shortest_routes_pairs, shortest_route_start_end):
        self.graph = graph
        self.start_node = start_node
        self.dest_node = dest_node
        self.preference_stations = preference_stations
        self.shortest_routes_start = shortest_routes_start
        self.shortest_routes_dest = shortest_routes_dest
        self.all_shortest_routes_pairs = all_shortest_routes_pairs
        self.shortest_route_start_end = shortest_route_start_end

    def create_new_graph(self):
        new_graph = nx.DiGraph()

        new_graph.add_nodes_from([self.start_node, self.dest_node] + self.preference_stations)

        for station in self.preference_stations:
            try:
                new_graph.add_edge(station, self.dest_node, weight=self.shortest_routes_dest[station][1])
            except KeyError:
                pass

        for station, (shortest_path, cumulative_weight) in self.shortest_routes_start.items():
            try:
                new_graph.add_edge(self.start_node, station, weight=cumulative_weight)
            except KeyError:
                pass

        for (station1, station2), (shortest_path, cumulative_weight) in self.all_shortest_routes_pairs.items():
            try:
                new_graph.add_edge(station1, station2, weight=cumulative_weight)
            except KeyError:
                pass
        
        if self.shortest_route_start_end is not None:
            new_graph.add_edge(self.start_node, self.dest_node, weight=self.shortest_route_start_end[1])

        return new_graph

class RouteWithWeights:
    def __init__(self, graph, shortest_routes_start, shortest_routes_dest, all_shortest_routes_pairs, shortest_routes_start_end):
        self.graph = graph
        self.shortest_routes_start = shortest_routes_start
        self.shortest_routes_dest = shortest_routes_dest
        self.all_shortest_routes_pairs = all_shortest_routes_pairs
        self.shortest_routes_start_end = shortest_routes_start_end

    def get_route_with_weights_start(self, start_node, target_node, mode):
        if target_node not in self.shortest_routes_start:
            print(f"No route found from {start_node} to {target_node}")
            return None, None

        path, total_weight = self.shortest_routes_start[target_node]
        route_with_weights = []
        
        for i in range(len(path) - 1):
            if path[i] not in self.graph or path[i + 1] not in self.graph[path[i]]:
                print(f"Edge not found in graph: {path[i]} -> {path[i + 1]}")
                continue
            edge_weight = self.graph[path[i]][path[i + 1]]['weight']
            route_with_weights.append((path[i], path[i + 1], mode, edge_weight))
        
        return route_with_weights, total_weight

    def get_route_with_weights_pairs(self, start_station, end_station, mode):
        if (start_station, end_station) not in self.all_shortest_routes_pairs:
            print(f"No route found between {start_station} and {end_station}")
            return None, None

        path, total_weight = self.all_shortest_routes_pairs[(start_station, end_station)]
        route_with_weights = []

        for i in range(len(path) - 1):
            if path[i] not in self.graph or path[i + 1] not in self.graph[path[i]]:
                print(f"Edge not found in graph: {path[i]} -> {path[i + 1]}")
                continue
            edge_weight = self.graph[path[i]][path[i + 1]]['weight']
            route_with_weights.append((path[i], path[i + 1], mode, edge_weight))
        
        return route_with_weights, total_weight

    def get_route_with_weights_dest(self, start_station, dest_node, mode):
        if start_station not in self.shortest_routes_dest:
            print(f"No route found from {start_station} to {dest_node}")
            return None, None

        path, total_weight = self.shortest_routes_dest[start_station]
        route_with_weights = []

        for i in range(len(path) - 1):
            if path[i] not in self.graph or path[i + 1] not in self.graph[path[i]]:
                print(f"Edge not found in graph: {path[i]} -> {path[i + 1]}")
                continue
            edge_weight = self.graph[path[i]][path[i + 1]]['weight']
            route_with_weights.append((path[i], path[i + 1], mode, edge_weight))
        
        return route_with_weights, total_weight

    def get_complete_route_with_weights(self, path_sequence):
        complete_route_with_weights = []

        for i in range(0, len(path_sequence), 2):
            start_node, intermediate_node, mode, _ = path_sequence[i]
            if i == 0:
                # First pair: start_node to the first preference station
                next_node = path_sequence[i + 2][0] if i + 2 < len(path_sequence) else path_sequence[i + 1][0]
                route, _ = self.get_route_with_weights_start(start_node, next_node, mode)
            elif i == len(path_sequence) - 1:
                # Last pair: last preference station to dest_node
                prev_node = path_sequence[i - 2][1] if i - 2 >= 0 else path_sequence[i - 1][1]
                breakpoint()
                route, _ = self.get_route_with_weights_dest(prev_node, start_node, mode)
                route = [(tup[1], tup[0], tup[2], tup[3],tup[4]) for tup in reversed(route)]
                breakpoint
            else:
                # Middle pairs: between preference stations
                prev_node = path_sequence[i - 2][1]
                next_node = path_sequence[i + 2][0] if i + 2 < len(path_sequence) else path_sequence[i + 1][0]
                route, _ = self.get_route_with_weights_pairs(prev_node, next_node, mode)
            
            if route:
                complete_route_with_weights.extend(route)

        return complete_route_with_weights

class RouteFinder:
    def __init__(self, graph, speed_dict):
        self.graph = graph
        self.speed_dict = speed_dict


            # edge_id = i
            # speeds = self.speed_dict[edge_id]
            
            # for s in set(self.node_stations[i]).intersection(self.node_stations[j]):
            #     if s != 'walk':
            #         # if  s not in self.user_preference:
            #         if s not in self.user_preference:
            #             self.costs[i, j, s] = edge_weight * 1e7
            #         else:
            #             if s == 'eb' or s == 'es':
            #                 speed = speeds['bike_speed']
            #             elif s == 'ec':
            #                 speed = speeds['car_speed']

            #             if speed == 0:
            #                 self.costs[i, j, s] = 1e7#float('inf')
            #             else:
            #                 self.costs[i, j, s] = edge_weight / (speed)

    def get_route_with_weights_start(self, start_node, shortest_routes_start, target_node, mode):
        if target_node not in shortest_routes_start:
            print(f"No route found from {start_node} to {target_node}")
            return None, None

        path, total_weight = shortest_routes_start[target_node]
        route_with_weights = []

        for i in range(len(path) - 1):
            if path[i] not in self.graph or path[i + 1] not in self.graph[path[i]]:
                print(f"Edge not found in graph: {path[i]} -> {path[i + 1]}")
                continue
            edge_weight = self.graph[path[i]][path[i + 1]]['weight']
            edge_speed = self.speed_dict[path[i]]['pedestrian_speed' if mode == 'walk'
                                                      else 'bike_speed' if mode == 'es' or mode == 'eb'
                                                      else 'car_speed'  if mode == 'ec' 
                                                      else 0  ]
            route_with_weights.append((path[i], path[i + 1], mode, edge_weight, edge_speed))

        return route_with_weights, total_weight

    def get_route_with_weights_pairs(self, all_shortest_routes_pairs, start_station, end_station, mode):
        if (start_station, end_station) not in all_shortest_routes_pairs:
            print(f"No route found between {start_station} and {end_station}")
            return None, None
        
        path, total_weight = all_shortest_routes_pairs[(start_station, end_station)]
        route_with_weights = []

        for i in range(len(path) - 1):
            if path[i] not in self.graph or path[i + 1] not in self.graph[path[i]]:
                print(f"Edge not found in graph: {path[i]} -> {path[i + 1]}")
                continue
            edge_weight = self.graph[path[i]][path[i + 1]]['weight']
            edge_speed = self.speed_dict[path[i]]['pedestrian_speed' if mode == 'walk'
                                                      else 'bike_speed' if mode == 'es' or mode == 'eb'
                                                      else 'car_speed' if mode == 'ec' 
                                                      else 0  ]
            route_with_weights.append((path[i], path[i + 1], mode, edge_weight, edge_speed))

        return route_with_weights, total_weight

    def get_route_with_weights_dest(self, dest_node, shortest_routes_dest, start_station, mode):
        if start_station not in shortest_routes_dest:
            print(f"No route found from {start_station} to {dest_node}")
            return None, None
        
        path, total_weight = shortest_routes_dest[start_station]
        route_with_weights = []

        for i in range(len(path) - 1):
            if path[i] not in self.graph or path[i + 1] not in self.graph[path[i]]:
                print(f"Edge not found in graph: {path[i]} -> {path[i + 1]}")
                continue
            edge_weight = self.graph[path[i]][path[i + 1]]['weight']
            edge_speed = self.speed_dict[path[i + 1]]['pedestrian_speed' if mode == 'walk'
                                                      else 'bike_speed' if mode == 'es' or mode == 'eb'
                                                      else 'car_speed' if mode == 'ec' 
                                                      else 0  ]
            route_with_weights.append((path[i], path[i + 1], mode, edge_weight, edge_speed))

        return route_with_weights, total_weight

    def get_complete_route_with_weights(self, path_sequence, shortest_routes_start, shortest_routes_dest, all_shortest_routes_pairs, shortest_route_start_end):
        complete_route_with_weights = []

        if len(path_sequence) == 1:
            start_node, end_node, mode, _ = path_sequence[0]
            path, total_weight = shortest_route_start_end

            for i in range(len(path) - 1):
                if path[i] not in self.graph or path[i + 1] not in self.graph[path[i]]:
                    print(f"Edge not found in graph: {path[i]} -> {path[i + 1]}")
                    continue
                edge_weight = self.graph[path[i]][path[i + 1]]['weight']
                edge_speed = self.speed_dict[path[i]]['pedestrian_speed' if mode == 'walk'
                                                      else 'bike_speed' if mode == 'es' or mode == 'eb'
                                                      else 'car_speed' if mode == 'ec' 
                                                      else 0  ]
                complete_route_with_weights.append((path[i], path[i + 1], mode, edge_weight, edge_speed))
        elif len(path_sequence) == 2:
            for i in range(len(path_sequence)):
                start_node, intermediate_node, mode, _ = path_sequence[i]
               
                if i == 0:
                    # First element: start_node to the first preference station
                    next_node = path_sequence[i + 1][0] if i + 1 < len(path_sequence) else None
                    if next_node:
                        route, _ = self.get_route_with_weights_start(start_node, shortest_routes_start, next_node, mode)
                elif i == len(path_sequence) - 1:
                    # Last element: last preference station to dest_node
                    prev_node = path_sequence[i - 1][1]
                    route, _ = self.get_route_with_weights_dest(prev_node, shortest_routes_dest, start_node, mode)
                    route = [(tup[1], tup[0], tup[2], tup[3], tup[4]) for tup in reversed(route)]
                else:
                    # Middle elements: between preference stations
                    prev_node = path_sequence[i - 1][1]
                    next_node = path_sequence[i + 1][0] if i + 1 < len(path_sequence) else None
                    if next_node:
                        route, _ = self.get_route_with_weights_pairs(all_shortest_routes_pairs, prev_node, next_node, mode)

                if route:
                    complete_route_with_weights.extend(route)
        else:
            for i in range(0, len(path_sequence)):
                start_node, intermediate_node, mode, _ = path_sequence[i]
                if i == 0:
                    # First pair: start_node to the first preference station
                    next_node = path_sequence[i + 2][0] if i + 2 < len(path_sequence) else path_sequence[i + 1][0]
                    route, _ = self.get_route_with_weights_start(start_node, shortest_routes_start, next_node, mode)
                elif i == len(path_sequence) - 1:
                    # Last pair: last preference station to dest_node
                    prev_node = path_sequence[i - 2][1] if i - 2 >= 0 else path_sequence[i - 1][1]
                    route, _ = self.get_route_with_weights_dest(prev_node, shortest_routes_dest, start_node, mode)
                
                    route = [(tup[0], tup[1], tup[2], tup[3], tup[4]) for tup in (route)]
                else:
                    # Middle pairs: between preference stations
                    if path_sequence[i][1] in ['walk', 'ec', 'eb', 'es']:
                        # print(f"Transition node skipping {i}")
                        continue
                    route, _ = self.get_route_with_weights_pairs(all_shortest_routes_pairs, path_sequence[i][0], path_sequence[i][1], mode)

                if route:
                    complete_route_with_weights.extend(route)

        return complete_route_with_weights


######################*Original Main for Single Pair*********************
##########################################################################
# Main code
file_path = 'DCC.net.xml'
speed_file_path = 'query_results-0.json'

# Create graph from XML file
graph_handler = GraphHandler(file_path)
original_G = graph_handler.get_graph()

# Define parameters
num_nodes = len(original_G.nodes)
station_types = ['eb', 'es', 'ec', 'walk']
node_stations = {i: station_types for i in original_G.nodes}
no_pref_nodes = 10
max_station_changes = 5
start_node = '-317554422' #'129211824'
start_station = 'walk'
end_node = '666320561#0'   #'25356998'
end_station = 'walk'
M = 1e6


# Generate prelferred station types for each node
preference_generator = PreferenceGenerator(original_G, station_types, no_pref_nodes)
preferred_station, preferred_nodes = preference_generator.generate_node_preferences()

# Compute shortest routes
shortest_path_computer = ShortestPathComputer(original_G)
shortest_routes_start = shortest_path_computer.compute_shortest_paths_start(start_node, preferred_nodes)
all_shortest_routes_pairs = shortest_path_computer.compute_shortest_paths_pairs(preferred_nodes)
shortest_routes_dest = shortest_path_computer.compute_shortest_paths_dest(end_node, preferred_nodes)
shortest_route_start_end = shortest_path_computer.compute_shortest_path_start_end(start_node,end_node)


# Create a new reduced graph
reduced_graph_creator = ReducedGraphCreator(original_G, start_node, end_node, preferred_nodes, shortest_routes_start, shortest_routes_dest, all_shortest_routes_pairs, shortest_route_start_end)
reduced_G = reduced_graph_creator.create_new_graph()

# Load speed data from JSON
with open(speed_file_path, 'r') as f:
    speed_data = json.load(f)

# Create a dictionary for speed data
speed_dict = {entry['edge_id']: {'pedestrian_speed': float(entry['pedestrian_speed']),
                                 'bike_speed': float(entry['bike_speed']),
                                 'car_speed': float(entry['car_speed'])}
              for entry in speed_data}

# User preferences
user_preference = ['eb', 'ec', 'es']

# Set up and solve the optimization problem
optimization_problem = OptimizationProblem(reduced_G, node_stations, preferred_station, M, speed_dict, user_preference)
optimization_problem.setup_decision_variables()
optimization_problem.setup_costs()
optimization_problem.setup_problem(start_node, start_station, end_node, end_station, max_station_changes)

# Solve the problem and measure execution time
prob, execution_time = optimization_problem.solve()

# Generate path sequence and output resultsq
if pulp.LpStatus[prob.status] == 'Optimal':
    print("Total Cost: ", pulp.value(prob.objective))

    path_finder = PathFinder(optimization_problem.paths, optimization_problem.station_changes, optimization_problem.costs,
                             optimization_problem.station_change_costs)
    path_sequence, station_change_count = path_finder.generate_path_sequence(start_node, start_station, end_node, end_station)

    print("Optimal Path from Reduced Graph:", path_sequence)
    print("Total Number of Station Changes:", station_change_count)
else:
    print("No optimal solution found.")

print("Execution time: {:.2f} seconds".format(execution_time))

# Instantiate the class with the required data
route_with_weights_handler = RouteFinder(
    original_G
    ,speed_dict
)

# Get the complete route with weights
complete_route_with_weights = route_with_weights_handler.get_complete_route_with_weights(path_sequence,shortest_routes_start,shortest_routes_dest,all_shortest_routes_pairs,shortest_route_start_end)
#Output
print("Mapped Path to original Graph with speeds:", complete_route_with_weights)


######################O*****************Original END *********************
##########################################################################


#################PipeLine to Execute the OD pairs from CSV ###################

# Main code
# file_path = 'DCC.net.xml'
# speed_file_path = 'query_results-0.json'
# od_pairs_file = 'random_od_pairs_200.csv'  # Path to the CSV file containing OD pairs
# output_csv_file = 'ReducedGraph_CBC_10Stations.csv'  # Output CSV file to store the results




# # solver = 'CBC' # Gurobi, GLPK
# # Create graph from XML file
# graph_handler = GraphHandler(file_path)
# original_G = graph_handler.get_graph()

# # Define parameters
# num_nodes = len(original_G.nodes)
# station_types = ['eb', 'es', 'ec', 'walk']
# node_stations = {i: station_types for i in original_G.nodes}
# no_pref_nodes = 20
# max_station_changes = 5
# M = 1e6

# # route_finder = RouteFinder(original_G)


# # Generate preferred station types for each node (execute only once)
# preference_generator = PreferenceGenerator(original_G, station_types, no_pref_nodes)
# preferred_station, preferred_nodes = preference_generator.generate_node_preferences()

# # Compute shortest routes pairs (execute only once)
# shortest_path_computer = ShortestPathComputer(original_G)
# precompute_time_start = time.time()
# all_shortest_routes_pairs = shortest_path_computer.compute_shortest_paths_pairs(preferred_nodes)
# precompute_time_end = time.time()

# precompute_time = precompute_time_end - precompute_time_start
# # Load speed data from JSON
# with open(speed_file_path, 'r') as f:
#     speed_data = json.load(f)

# # Create a dictionary for speed data
# speed_dict = {entry['edge_id']: {'pedestrian_speed': float(entry['pedestrian_speed']),
#                                  'bike_speed': float(entry['bike_speed']),
#                                  'car_speed': float(entry['car_speed'])}
#               for entry in speed_data}

# # User preferences
# user_preference = ['eb','ec','es']  # This has to be Default - Remove any if not 

# # Prepare the output CSV file
# with open(output_csv_file, 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(['Start Node', 'End Node', 'Total Cost', 'Execution Time', 'Optimal Path Sequence', 'pre_compute_time'])

#     # Read OD pairs from CSV file and execute the main loop
#     with open(od_pairs_file, 'r') as odfile:
#         reader = csv.reader(odfile)
#         next(reader)  # Skip header row if present
#         for row in reader:

#             initial_time = time.time()

#             start_node, end_node = row  # Extract start_node and end_node from the current row

           
#             # Compute shortest routes start
#             shortest_routes_start = shortest_path_computer.compute_shortest_paths_start(start_node, preferred_nodes)

#             # Compute shortest routes dest
#             shortest_routes_dest = shortest_path_computer.compute_shortest_paths_dest(end_node, preferred_nodes)

#             #compute shortest route between start and end 
#             shortest_route_start_end = shortest_path_computer.compute_shortest_path_start_end(start_node, end_node)
#             # Create a new reduced graph
#             reduced_graph_creator = ReducedGraphCreator(original_G, start_node, end_node, preferred_nodes, shortest_routes_start, shortest_routes_dest, all_shortest_routes_pairs,shortest_route_start_end)
#             reduced_G = reduced_graph_creator.create_new_graph()

#             # Set up and solve the optimization problem
#             optimization_problem = OptimizationProblem(reduced_G, node_stations, preferred_station, M, speed_dict, user_preference)
#             optimization_problem.setup_decision_variables()
#             optimization_problem.setup_costs()
#             optimization_problem.setup_problem(start_node, 'walk', end_node, 'walk', max_station_changes)

#             try:
#                 # Solve the problem and measure execution time
#                 prob, execution_time = optimization_problem.solve()

#                 if pulp.LpStatus[prob.status] == 'Optimal':
#                     total_cost = pulp.value(prob.objective)

#                     path_finder = PathFinder(optimization_problem.paths, optimization_problem.station_changes, optimization_problem.costs,
#                                              optimization_problem.station_change_costs)
#                     path_sequence, station_change_count = path_finder.generate_path_sequence(start_node, 'walk', end_node, 'walk')


#                     end_time = time.time()
#                     Total_time = end_time - initial_time


                    
#                     # route_with_weights = route_finder.get_complete_route_with_weights(path_sequence, shortest_routes_start, shortest_routes_dest, all_shortest_routes_pairs, shortest_route_start_end)
#                     # Write results to CSV
#                     writer.writerow([start_node, end_node, total_cost, Total_time, path_sequence, precompute_time])
#                 else:
#                     # Write results to CSV with 'inf' for total cost if no optimal solution is found
#                     writer.writerow([start_node, end_node, 'inf', 'No optimal solution found'])
#             except pulp.PulpSolverError:
#                 # Write results to CSV with 'inf' for total cost if solver fails
#                 writer.writerow([start_node, end_node, 'inf', 'Solver failed'])
