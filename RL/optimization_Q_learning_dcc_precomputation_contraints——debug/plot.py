import matplotlib.pyplot as plt


class Plotter:
    def __init__(self):
        self.x = [100, 150, 200, 250, 300]
        self.y = [0.17331409454345703, 0.2820098400115967, 0.36731696128845215, 0.47672295570373535, 0.5482101440429688]
        self.z = [6615.158972839505, 3966.5241492119585, 5141.417344178863, 6100.607879835392, 6083.433073529574]

    def plot(self):
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Set labels for axes
        ax.set_xlabel('Number of Ants')
        ax.set_ylabel('Execution Time (seconds)')
        ax.set_zlabel('Time Cost (seconds)')

        # Plotting the data points
        ax.scatter(self.x, self.y, self.z, color='red', s=100, label='ACO Performance')
        plt.show()


if __name__ == '__main__':
    plotter = Plotter()
    plotter.plot()
