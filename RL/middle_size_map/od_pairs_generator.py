import csv
from user_info import User
from optimization import Optimization


def generate_od_pairs():
    net_xml_path = './medium/graph202311211622.net.xml'
    source_edg = '-171'
    target_edg = '-174'
    start_mode = 'walking'
    ant_num = [350]
    episodes = [500, 1000, 1500, 2000]
    iteration = 1
    db_path = 'test_new.db'
    user = User(60, True, 0, 20)

    optimization = Optimization(net_xml_path, user, db_path, source_edg, target_edg)
    od_pairs = optimization.choose_od_pairs()

    with open('od_pairs_50.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for pair in od_pairs:
            writer.writerow(pair)  


if __name__ == '__main__':
    generate_od_pairs()
