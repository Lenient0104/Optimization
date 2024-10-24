import pandas as pd
import matplotlib.pyplot as plt



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
    filtered_data = data[(data['Find'] != False) & (data['Initial Energy'] != 0.01)]

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
    unique_ants = filtered_data['Initial Energy'].unique()


    # Prepare data for plotting
    # all_data = [filtered_data[filtered_data['Number of Ants'] == ant]['Travel Time Cost (seconds)'].dropna() for ant in
    #             unique_ants]
    all_data = [filtered_data[filtered_data['Initial Energy'] == ant]['Travel Time Cost (seconds)'].dropna() for ant in
                unique_ants]
    plt.rcParams.update({
        'font.size': 18,  # Global font size
        'axes.titlesize': 20,  # Title font size
        'axes.labelsize': 18,  # X and Y axis labels font size
        'xtick.labelsize': 18,  # X tick labels font size
        'ytick.labelsize': 18,  # Y tick labels font size
        'legend.fontsize': 18,  # Legend font size
        'figure.titlesize': 22  # Figure title font size
    })
    # Plot box plot
    plt.figure(figsize=(10, 6))
    plt.boxplot(all_data, labels=unique_ants)
    plt.xlabel('Initial Energy')
    plt.ylabel('Travel Time Cost (seconds)')
    # plt.title('ACO Box Plot of Travel Time Cost for Different Initial Energy')
    plt.savefig('Q Learning Box Plot of Travel Time Cost 8.eps', format='eps')
    plt.show()


def plot_box_plot_execution_time(filtered_data):
    """
    Plot box plot for Execution Time.

    Args:
    filtered_data (pandas DataFrame): Filtered DataFrame.
    """
    # Get unique number of ants
    # unique_episodes = filtered_data['Episode'].unique()
    unique_ants = filtered_data['Initial Energy'].unique()

    # Prepare data for plotting
    # all_data = [filtered_data[filtered_data['Number of Ants'] == ant]['Travel Time Cost (seconds)'].dropna() for ant in
    #             unique_ants]
    all_data = [filtered_data[filtered_data['Initial Energy'] == ant]['Execution Time (seconds)'].dropna() for ant in
                unique_ants]

    # Plot box plot
    plt.rcParams.update({
        'font.size': 18,  # Global font size
        'axes.titlesize': 20,  # Title font size
        'axes.labelsize': 18,  # X and Y axis labels font size
        'xtick.labelsize': 18,  # X tick labels font size
        'ytick.labelsize': 18,  # Y tick labels font size
        'legend.fontsize': 18,  # Legend font size
        'figure.titlesize': 22  # Figure title font size
    })
    plt.figure(figsize=(10, 6))
    plt.boxplot(all_data, labels=unique_ants)
    plt.xlabel('Initial Energy')
    plt.ylabel('Execution Time (seconds)')
    # plt.title('ACO Box Plot of Execution Time for Different Initial Energy')
    plt.autoscale()
    plt.savefig('Q Learning Box Plot of Execution Time 8.eps', format='eps')
    plt.show()


# Call the functions
filtered_data = filter_data("qLearning.csv")
plot_box_plot_travel_time(filtered_data)
plot_box_plot_execution_time(filtered_data)
