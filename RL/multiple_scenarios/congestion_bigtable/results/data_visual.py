import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def filter_data(csv_file):
    """
    Filter out rows where Travel Time Cost is infinity.

    Args:
    csv_file (str): Path to the CSV file.

    Returns:
    pandas DataFrame: Filtered DataFrame.
    """
    # Read the CSV file
    data = pd.read_csv(csv_file)

    # Filter out rows where Travel Time Cost is not False
    filtered_data = data[data['Find'] != False]

    data['Find'] = data['Find'].replace({'TRUE': True, 'FALSE': False})

    # 保存修改后的表格
    data.to_csv(csv_file, index=False)

    print(filtered_data)

    return filtered_data


def plot_box_plot_travel_time(filtered_data):
    """
    Plot box plots for each unique number of ants.

    Args:
    filtered_data (pandas DataFrame): Filtered DataFrame.
    """
    # Get unique number of ants
    # unique_episodes = filtered_data['Episode'].unique()
    unique_ants = filtered_data['Simulation Time'].unique()

    # Prepare data for plotting
    # all_data = [filtered_data[filtered_data['Number of Ants'] == ant]['Travel Time Cost (seconds)'].dropna() for ant in
    #             unique_ants]
    all_data = [filtered_data[filtered_data['Simulation Time'] == ant]['Travel Time Cost (seconds)'].dropna() for ant in
                unique_ants]

    # Plot box plot
    plt.figure(figsize=(12, 8))
    plt.boxplot(all_data, labels=unique_ants)
    plt.xlabel('Simulation Time')
    plt.ylabel('Travel Time Cost (seconds)')
    plt.title('DQN: Box Plot of Travel Time Cost for Different Simulation Times')
    plt.savefig('DQN: Box Plot of Travel Time Cost')
    plt.show()


def plot_box_plot_execution_time(filtered_data):
    """
    Plot box plot for Execution Time.

    Args:
    filtered_data (pandas DataFrame): Filtered DataFrame.
    """
    # Get unique number of ants
    # unique_episodes = filtered_data['Episode'].unique()
    unique_ants = filtered_data['Number of Ants'].unique()

    # Prepare data for plotting
    # all_data = [filtered_data[filtered_data['Episode'] == episode]['Execution Time (seconds)'].dropna() for episode in
    #             unique_episodes]
    all_data = [filtered_data[filtered_data['Number of Ants'] == ant]['Execution Time (seconds)'].dropna() for ant in
                unique_ants]

    # Plot box plot
    plt.figure(figsize=(12, 8))
    plt.boxplot(all_data, labels=unique_ants)
    plt.xlabel('Number of Ants')
    plt.ylabel('Execution Time (seconds)')
    plt.title('ACO: Box Plot of Execution Time for Different Numbers of Episodes')
    plt.savefig('ACO: Box Plot of Execution Time')
    plt.show()


# Call the functions
filtered_data = filter_data("DQN_experiment_results_simulation_2000_episode")
plot_box_plot_travel_time(filtered_data)
# plot_box_plot_execution_time(filtered_data)
