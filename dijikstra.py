import json
import networkx as nx
import xml.etree.ElementTree as ET
import re
import csv
from tqdm import tqdm
import heapq
import time
import random

DEFAULT_SPEEDS = {
    'e-bike': 1.5,  # Default speed of e-bike
    'e-car': 10.5,  # Default speed of e-car
    'e-scooter': 5.5,  # Default speed of e-scooter
    'walking': 1.5  # Default speed of walking
}

# 10
E_HUBS = ['361409608#3', '3791905#2', '-11685016#2', '369154722#2', '244844370#0', '37721356#0', '74233405#1',
          '129774671#0', '23395388#5', '-64270141']  # random.sample(list(self.G.nodes), self.num_preferred_nodes)
# 20 E_HUBS = ['361409608#3', '3791905#2', '-11685016#2', '369154722#2', '244844370#0', '37721356#0','74233405#1',
# '129774671#0', '23395388#5', '-64270141', '18927706#0', '-42471880', '67138626#1', '41502636#0', '-75450412',
# '-23347664#1', '14151839#3', '-12341242#1', '-13904652','-47638297#3'] 50 E_HUBS = ['-52881219#6', '3788262#0',
# '-36819763#3', '151482073', '-44432110#1', '-45825602#2', '-37721078#5', '-4834964#1', '59433202', '-441393227#1',
# '20485040#0', '29333277', '-38062537', '37812705#0', '3708641#0', '4934504#3', '-8232242#0', '254372188#2',
# '-114208220#0', '-370381577#2', '228694841', '-22716429#2', '129068838', '-191042862#1', '42636056', '7849019#2',
# '556290239#1', '-63844803#0', '303325200#2', '4395609#0', '-575044518', '77499019#1', '-38375201#6', '129211829#0',
# '-228700735', '23086918#0', '-4825259#1', '-481224216#4', '-5825856#7', '-7989926#3', '4385842#0', '30999830#4',
# '529166764#0', '334138114#8', '-326772508#1', '3708865#8', '25787652', '55668420#1', '55814730#3', '-561862256']
# 100 E_HUBS = ['128521563#0', '401904747', '31897169#3', '3284827#0', '-52241896', '372838330#2', '-376465145#1',
# '432565295#5', '24644260#0', '4540245#3', '4934643#0', '25536089#1', '-67139161#4', '30988145#0', '-282195032#5',
# '-48205720#2', '-128450676#3', '-115656107', '75741007', '32346509', '3818299#3', '-128946815#7', '-111746918',
# '38864100#0', '31350775#4', '662531575#0', '-43870791#1', '44385285', '-44436781#1', '129211827#0', '94838278#4',
# '-43175828#0', '37747500#5', '-129211821#4', '-25761629', '125813654#1', '369359214', '369154720#2', '30583814#1',
# '-365945819#0', '4541293#0', '520012239#0', '81982578#10', '-7919284#10', '7919276#0', '-326874928#1',
# '147077493#0', '45619558', '-127220738', '-14047195', '-45530928#0', '-372245390#1', '23501248#4', '14047190#0',
# '-26150087#3', '-7977648#1', '55668414#1', '256024070#3', '-129211821#7', '3191572#5', '-369977727#1',
# '677070805#4', '-3191572#1', '553868718#1', '-48226178#0', '-369622780', '-20848460#1', '174092689#0', '228694841',
# '64270037#0', '4385874#1', '-238740928#2', '45530928#1', '-80478293#2', '432565295#3', '-317027066#0',
# '-3818568#3', '-39247679#3', '285487940', '-4540248#0', '-3708538#2', '-129211838', '231340188#0', '-13998367',
# '-3789251#3', '6653000#5', '317219094#1', '48454248#0', '13878225#0', '-495995815', '-94838278#4', '-38090686#2',
# '32201932', '-30454721#1', '-147463637#1', '-49728145#0', '143866317#0', '-14039614', '-636824321', '-228665722']
# 200 E_HUBS =['-14047498#3', '128676689#5', 'gneE25', '-3788879#2', '3266498', '-7919280#1', '111286353#2',
# '4834963#1', '3791210#1', '7919275#3', '-4825239#1', '497486086', '-44409757#0', '39540888#1', '283633789',
# '4395858#1', '-37721078#6', '14047759#0', '8045273#0', '-45406018', '45406017', '-222609066#1', '-32016067#0',
# '-44705513', '-13866422', '-300664821#4', '-394233223#2', '29561733#0', '-362985327#2', '-41592110', '317554424#4',
# '-319556828#0', '41592114', '-4215597#5', '230478913', '4396062#0', '137621708#0', '-4825332', '317101094#12',
# '372256216#3', '-28002618#0', '4934662#8', '71423403#2', '3708534#0', '-30999712', '-49728136#1', '-20848457',
# '30655095#4', '231612950#0', '-3708465#1', '37747500#3', '14039609#0', '14047769#0', '-44404539#1', '-3191573#4',
# '41851537#1', '37746962', '40047164#0', '-7919286#1', '16247623', '37722081#0', '32631306#0', '55668395#0',
# '121249650#6', '362939722', '3708878#1', '-38574544#2', '3763679#1', '-4934662#7', '114207514#0', '52241890#1',
# '13904682#3', '41502636#0', '-114277935#1', '-51128241', '-114208222#0', '-44432110#0', '3788877#0', '-370154355',
# '-49728142#2', '-4691594', '-115656107', '30389516#0', '-3191572#5', '4395933#6', '-46895350#2', '13904652',
# '128450677#2', '-405364257#0', '129211857#3', '-308731562#1', '-4475686#2', '-4264251', '375581292#0', '362939725',
# '530413782', '-40701185', '23011304#1', '-23076566', '4287188#2', '-310303150', '-3266486#10', '-3708865#5',
# '4937390', '6273610#2', '-317697975#0', '-8384431#7', '-38122989', '-405546346', '-145344471#0', '-31008788',
# '3708468', '-22566801#0', '13871411', '-23011304#1', '-3936651', '4934682#0', '529166766#0', '379071737#0',
# '-282195032#5', '43316112#0', '-7977023#4', '-48985672', '-55814729#3', '13904645#0', '-30047545#3', '13878308#0',
# '-37721078#4', '31992229#1', '4919615#4', '-4571527#3', '3791745#4', '20848454#2', '-30999830#0', '37692199#0',
# '18928030#0', '110407380#0', '481221025', '289876870', '32016067#1', '-140673640', '-317145973#0', '41772640',
# '-334138117#0', '-38864096#1', '368505007#2', '23395388#5', '125813655#0', '-10892467#0', '-4934446#0',
# '77499019#1', '-227677863#2', '-31897170', '317101094#7', '25761610#0', '25535969#2', '4401701#1', '7421646#8',
# '125863788', '-185127488#2', '29561733#3', '-24234804', '4395994#0', '-348538605#0', '-67139161#4', '20848425#0',
# '-7989949#1', '-375581292#0', '-3708538#1', '23567535#2', '-48520199', '2689742', '-18928030#2', '430591268#3',
# '-7919276#4', '25731437', '24776090#1', '-80439756#2', '317219091#3', '-3708543', '143866317#0', '-228731431#0',
# '481221028#4', '-3788877#4', '-376394516#0', '423283956#0', '-32117786#4', '44069390#3', '-3285420#2',
# '38574621#1', '4571527#0', '-23567536#1', '37692490#3', '4899745#0', '13871250', '129211835#0', '282195032#6',
# '-38122987#2', '-10575480#2', '-227767347#1'] 300 E_HUBS =['-528915436#0', '-683418232#1', '-13904703',
# '368505008#3', '4825258#0', '129211847#0', '40047164#0', '4834969#0', '-368505008#3', '-31992438#0', '-77499018',
# '-61717563#0', '233718298#1', '8232242#7', '282243997', '-20848429', '-394233223#2', '33238658', '-20686243',
# '-3785444#0', '-22978740#8', '191042860#0', '-3788879#5', '7919278#0', '658163878#0', '302527226#0', '44069392',
# '238740923#0', '-31008788', '45406018', '-231620391#1', '-44402155', '32491867#1', '-4934446#0', '125880830#9',
# '8468553#4', '-4395609#3', '-5825858#4', '-334138114#6', '-3788308#3', '38668910#2', '-366887650#2', '-31982174#1',
# '-129211812#7', '-32642508#4', '-31350775#5', '-231612455#3', '37692199#0', '14047764#0', '-132759854#2',
# '334138117#1', '222609066#2', '135583247', '-37692198#1', '59433202', '-14047194#0', '-547766101#0', '4545229#0',
# '-67139161#4', '4919460', '-110545289#2', '52375061', '-7919286#1', '12341232#0', '30047536#0', '-3785494#2',
# '348538605#1', '52241892', 'gneE18', '38669011#0', '30655092#0', '100654686', '-30655096', '-191042862#1',
# '372838330#0', '129211853', '-54451008#3', '228729058', '368646813#11', '369154717#1', '317219088', '-37747137#1',
# '60521432', '386571734#1', '-156375816#2', '191042863', '11685019#0', '-228557885#0', '10575471#2', '-3791210#1',
# '-238740928#2', '-31982160#2', '32491867#2', '-41592113#0', '-300664821#4', '4926030', '-22566802#0', '80479201#0',
# '38574544#2', '7421646#5', '-4934444#2', '362985327#0', '-228731434', '3789571#1', '-30454725#3', '-43321718#2',
# '129217471#0', '49961187#0', '371391166#0', '13904648#0', '4395692', '-532427444#3', '64272633#1', '-4396062#6',
# '-29333723#3', '379666708', '44007601#0', '30047545#2', '3789251#0', '23567534#7', '41772640', '48455143',
# '401912432#2', '230751238', '-317697974', '-375758262#1', '227711231#0', '-39523885#9', '4571527#4', '14047962#0',
# '-20485040#4', '23920340', '30047549#2', '-30046890', '-32015727', '-30454721#5', '-51128240#5', '52375066#1',
# '48455139#0', '-47861482#1', '64270035#0', '111746918', '40260272#0', '13904649#1', '-8384431#1', '-5825858#1',
# '-63919918', '4395994#0', '30999830#0', '2110698#0', '13904645#0', '3785449#1', '-32631772#2', '-4078065#4',
# '283771398#1', '30999836#1', '-54451007#3', '49728145#2', '48205721#2', '29554114#0', '-4287395#0', '143866317#0',
# '94838285#1', '162745568#0', '30655095#5', '-4934504#3', '-260607183#1', '-48455138', '31350775#5', '4541293#0',
# '-64344154', '-228818578', '64270141', '-13894939#5', '37747008#0', '-8384431#3', '-4829724#2', '-285991276',
# '-360342135#0', '-50503783#1', '54451008#3', '24601280', '8468553#3', '-375581291#1', '-100654687', '-7977647#1',
# '129211832', '63919918', '-10575471#0', '-7415911#3', '55814737', '22716429#1', '14039611#1', '4571776#7',
# '58403835#0', '31982229#0', '37692391#0', '-23395388#6', '56681848#2', '38668910#0', '49728141', '4628794#0',
# '-49728136#2', '-30988144#1', '-30818795#3', '-370154352#0', '229143734', '32117788#9', '3791207#0', '4215597#0',
# '673136293', '4215601#0', '-317697975#2', '3789251#4', '-423711652#1', '129211824', '22978622#0', '-24776091',
# '-4934446#2', '-130277328#0', '-317145978', '-3708548#1', '-39117802#1', '-317101094#11', '520012241#0',
# '-116689248#1', '27219884#2', '14047501#0', '80439756#0', '-372256216#4', '-593087259#1', '52881850#0',
# '3266486#2', '-4541293#8', '80479201#2', '29561733#3', '23307145#0', '44409754', '-29266714#0', '4287295#0',
# '372245388#0', '3788281#0', '306917251#1', '-476166088', '-41362435', '-30999836#0', '-3708680#2', '317544589#2',
# '23050596', '-13871230#1', '-137621706#0', '372460213#3', '-3708471#2', '-13904650#6', '94838282', '47861482#0',
# '-48996164', '581774894#1', '-4926120#3', '423711652#1', '4078079#1', '94838279', '-30454724', '-30046887#0',
# '129211813#0', '39117800#1', '4287395#2', '83806426#4', '-37721078#5', '-3285422#0', '302617608#1', '14327903#3',
# '30999830#4', '33854097#2', '129211797#2', '-31897169#3', '13998368#0', '-231612950#2', '30454721#0',
# '-44069390#2', '-228557873', '37691265#0', '-4934627', '-279585244#0', '13995755#0', '24450827', '4263428#2',
# '-38062538', '-481221027#2', '50502873', '-23076537#2', '-37721078#4', '4919615#0', '348538605#0', '32631306#0',
# '-3708547', '-26150087#2', '-331870104#4', '-231630881#1', '-191042861']

MINIMUM_SPEED = 0.00001  # A very small speed to avoid division by zero


def load_speed_data(json_file):
    with open(json_file, 'r') as file:
        speed_data = json.load(file)
    speed_map = {}
    for record in speed_data:
        edge_id = record['edge_id']
        speed_map[edge_id] = {
            'walking': float(record.get('pedestrian_speed', DEFAULT_SPEEDS['walking'])),
            'e-bike': float(record.get('bike_speed', DEFAULT_SPEEDS['e-bike'])),
            'e-car': float(record.get('car_speed', DEFAULT_SPEEDS['e-car'])),
            'e-scooter': float(record.get('bike_speed', DEFAULT_SPEEDS['e-scooter']))
        }
    return speed_map


# Function to create graph from .net.xml file
def create_graph_from_net_xml(file_path, preferred_modes):
    unique_edges = []
    connections = []
    pattern = r"^[A-Za-z]+"
    tree = ET.parse(file_path)
    root = tree.getroot()

    edge_detail_map = {}
    for edge in tqdm(root.findall('edge'), desc="Processing edges"):
        for lane in edge.findall('lane'):
            edge_id = edge.attrib['id']
            edge_detail_map[edge_id] = {
                'length': float(lane.attrib['length']),
                'speed_limit': float(lane.attrib['speed']),
                'shape': lane.attrib['shape'],
                # Modes are assigned based on whether the node is an e-hub or not
                'modes': preferred_modes if edge_id in E_HUBS else ['walking']
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
            modes = edge_detail_map[from_edge]['modes']  # Including modes in the edge
            G.add_edge(from_edge, to_edge, length=length, modes=modes)

    return G


class MultimodalGraph:
    def __init__(self, graph, e_hubs, speed_map, preferred_modes=None):
        self.graph = graph
        self.e_hubs = e_hubs
        self.speed_map = speed_map
        self.assign_modes_to_e_hubs(preferred_modes)

    def assign_modes_to_e_hubs(self, preferred_modes=None):
        """Assign e-mobility modes to e-hubs."""
        for hub in self.e_hubs:
            if preferred_modes:
                # Create a copy of preferred_modes without 'walking'
                modes_to_sample = [mode for mode in preferred_modes if mode != 'walking']
            else:
                # If preferred_modes is None, use the default list
                modes_to_sample = ['e-bike', 'e-scooter', 'e-car']

            # Randomly select a subset of modes
            available_modes = random.sample(modes_to_sample, k=random.randint(1, len(modes_to_sample)))
            available_modes.append('walking')  # Walking is always available
            self.graph.nodes[hub]['modes'] = available_modes

    def get_available_modes(self, node):
        """Get available modes at a given node."""
        return self.graph.nodes[node].get('modes', ['walking'])

    def calculate_time(self, distance, mode, edge):
        """Calculate time taken to traverse an edge using a specific mode."""
        speed = self.speed_map.get(edge, {}).get(mode, DEFAULT_SPEEDS[mode])

        if speed == 0:
            speed = MINIMUM_SPEED

        return distance / speed

    def can_transition(self, current_mode, current_hub, next_hub):
        """Check if we can transition from current_mode at current_hub to next_hub."""
        current_modes = self.get_available_modes(current_hub)
        next_modes = self.get_available_modes(next_hub)

        # We can only transition if the next hub supports the current mode
        return current_mode in next_modes and current_mode in current_modes

    def shortest_path(self, start, end, max_transitions):
        """Compute the shortest path from start to end considering multimodal options."""
        pq = [(0, start, 'walking', 0, [])]  # (time, current_node, current_mode, transitions, path)
        visited = set()

        while pq:
            current_time, current_node, current_mode, transitions, path = heapq.heappop(pq)

            if current_node in visited:
                continue

            visited.add(current_node)
            path = path + [(current_node, current_mode, current_time)]

            if current_node == end:
                # Format the path correctly
                formatted_path = []
                total_time_spent = current_time
                for i in range(1, len(path)):
                    prev_node, prev_mode, prev_time = path[i - 1]
                    curr_node, curr_mode, curr_time = path[i]
                    travel_time = curr_time - prev_time
                    formatted_path.append((prev_node, curr_node, prev_mode, travel_time))
                return formatted_path, total_time_spent

            for neighbor in self.graph.successors(current_node):
                if neighbor not in visited:
                    edge_data = self.graph[current_node][neighbor]
                    distance = edge_data['length']

                    for mode in self.get_available_modes(neighbor):
                        if mode == 'walking' or self.can_transition(current_mode, current_node, neighbor):
                            # Check if we can transition to the new mode
                            new_transitions = transitions + (1 if mode != current_mode else 0)

                            if new_transitions <= max_transitions:
                                travel_time = self.calculate_time(distance, mode, current_node)
                                heapq.heappush(pq, (current_time + travel_time, neighbor, mode, new_transitions, path))

        return None, None  # No path found


# Now process the O-D pairs
def process_od_pairs(graph, speed_map, input_csv, output_csv, max_transitions, preferred_modes):
    # Use the correct instantiation here
    multimodal_graph = MultimodalGraph(graph, E_HUBS, speed_map, preferred_modes=preferred_modes)

    with open(input_csv, 'r') as infile, open(output_csv, 'w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        writer.writerow(['Origin', 'Destination', 'Total Cost', 'Execution Time', 'Optimal Path Sequence'])
        for row in reader:
            origin, destination = row[0], row[1]
            st_time = time.time()
            path, total_time_spent = multimodal_graph.shortest_path(origin, destination,
                                                                    max_transitions)  # Example max_transitions
            end_time = time.time()
            exec_time = (end_time - st_time) * 1000
            if path:
                path_str = '; '.join([f"({p[0]}, {p[1]}, {p[2]}, {p[3]:.2f})" for p in path if len(p) == 4])
                writer.writerow([origin, destination, total_time_spent, exec_time, path_str])
            else:
                writer.writerow([origin, destination, "No path found"])
    return path


# Example usage
file_path = 'DCC.net.xml'
input_csv = 'od_pairs_sing.csv'
output_csv = 'low_traffic_demo_test.csv'
json_file = 'query_results-low_traffic1800.json'

max_transitions = 5
# Load speed data from JSON
speed_map = load_speed_data(json_file)
preferred_modes = ['walking', 'e-bike', 'e-car', 'e-scooter']
# Generate the graph from the .net.xml file
G = create_graph_from_net_xml(file_path, preferred_modes)

# Process the O-D pairs and write the output
path = process_od_pairs(G, speed_map, input_csv, output_csv, max_transitions, preferred_modes)
print(path)

