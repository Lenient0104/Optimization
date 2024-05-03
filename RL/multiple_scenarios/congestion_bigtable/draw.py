import matplotlib.pyplot as plt
import json

def plot_speed_boxplot(json_files, vehicle_type):
    speeds = []

    for json_file in json_files:
        with open(json_file, 'r') as f:
            query_results = json.load(f)

        vehicle_speeds = []

        for result in query_results:
            speed = result[f'{vehicle_type}_speed']
            vehicle_speeds.append(float(speed))

        speeds.append(vehicle_speeds)

    plt.figure(figsize=(10, 6))
    boxprops = dict(linestyle='-', linewidth=2, color='b')
    whiskerprops = dict(linestyle='--', linewidth=2, color='black')

    # 绘制箱线图
    bp = plt.boxplot(speeds, labels=simulation_times, whiskerprops=whiskerprops, boxprops=boxprops, showfliers=True)

    plt.title(f'Boxplot of {vehicle_type.capitalize()} Speeds at Different Simulation Times')
    plt.xlabel('Simulation Time')
    plt.ylabel(f'{vehicle_type.capitalize()} Speed')
    plt.xticks(rotation=45)

    # 检查四分位数是否相等
    for line in bp['boxes']:
        x, y = line.get_xydata()[0] # 获取箱体的底部坐标
        top = line.get_xydata()[1][1] # 获取箱体的顶部坐标
        if y == top: # 如果箱体的顶部和底部坐标相同
            plt.text(x, y, 'Very low variability', horizontalalignment='center', verticalalignment='center')

    plt.tight_layout()
    plt.savefig(f'{vehicle_type}_speed_boxplot.png')
    plt.show()

# Plot boxplots for each vehicle type
simulation_times = [0, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000]
json_files = [f'query_results-{sim}.json' for sim in simulation_times]
vehicle_types = ['pedestrian', 'bike', 'car']

