import random
import uuid


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
                           '-13904652', '-47638297#3']

        # 假设车辆种类有三种：ec（电动汽车）、es（电动滑板车）、eb（电动自行车）
        vehicle_types = ['ec', 'es', 'eb']
        vehicle_counts = {'ec': 5, 'es': 10, 'eb': 7}  # 假设每种车辆的初始数量
        vehicle_battery_range = (0.2, 1)  # 电量范围为 20% 到 100%
        initial_energy = {  # in wh
            'eb': 50,
            'es': 50,
            'ec': 3000,
            'walk': 0
        }

        vehicle_id_counter = 0  # 用于生成全局唯一的ID

        for i in self.G.nodes:
            if i in preferred_nodes:
                # 为每个站点生成对应的station_types
                preferred_types = ['ec', 'es', 'eb', 'walk']  # 站点的服务类型

                # 生成该站点的车辆和电量信息
                vehicles = []
                for v_type in vehicle_types:
                    # 为每种车辆生成一些具体数量的车辆以及它们的电量
                    num_vehicles = vehicle_counts[v_type]  # 假设每个站点的车辆数量是固定的
                    for _ in range(num_vehicles):
                        battery_level = random.uniform(*vehicle_battery_range) * initial_energy[v_type]  # 随机生成电量

                        # 生成全局唯一车辆ID（可以使用计数器或UUID）
                        vehicle_id_counter += 1
                        vehicle_id = f"{v_type}_{vehicle_id_counter}"  # 例如：'ec_1', 'es_2'

                        # 也可以使用uuid生成全局唯一的标识符
                        # vehicle_id = str(uuid.uuid4())  # 生成唯一ID

                        # 为每辆车创建字典，包含type、battery、id
                        vehicles.append({
                            'id': vehicle_id,  # 车辆唯一ID
                            'type': v_type,  # 车辆类型
                            'battery': battery_level  # 车辆电量
                        })

                # 添加站点的车辆信息
                preferred_station[i] = {
                    'types': preferred_types,
                    'vehicles': vehicles
                }

                # 确保 walk 是站点的类型之一
                if 'walk' not in preferred_station[i]['types']:
                    preferred_station[i]['types'].append('walk')

                # 将站点信息保存到 node_station_pair
                self.node_station_pair[i] = preferred_station[i]
            else:
                preferred_station[i] = {'types': [''], 'vehicles': []}  # 没有车辆信息的站点

        return preferred_station, preferred_nodes
