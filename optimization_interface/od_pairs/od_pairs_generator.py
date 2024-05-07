import csv
from optimization_interface.user_info import User
from optimization_interface.optimization import Optimization


def generate_od_pairs():
    net_xml_path = '../../optimization_interface/DCC.net.xml'
    source_edge = '361450282'
    target_edge = "-110407380#1"
    start_mode = 'walking'
    ant_num = [350]
    episodes = [500, 1000, 1500, 2000]
    iteration = 1
    db_path = 'test_new.db'
    user = User(60, True, 0, 20)

    optimization = Optimization(net_xml_path, user, db_path, 10000, source_edge, target_edge)
    od_pairs = optimization.choose_od_pairs()

    with open('od_pairs_500_new.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for pair in od_pairs:
            writer.writerow(pair)  


if __name__ == '__main__':
    generate_od_pairs()
