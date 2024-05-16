def calculate_energy_comsumption(self, current_mode, distance):
    if current_mode == 'walking':
        return 0
    # Define vehicle efficiency in Wh per meter (converted from Wh per km)
    vehicle_efficiency = {'e_bike_1': 20 / 1000, 'e_scooter_1': 25 / 1000, 'e_car': 150 / 1000}
    # battery_capacity = {'e_bike_1': 500, 'e_scooter_1': 250, 'e_car': 50000}
    battery_capacity = {'e_bike_1': 500, 'e_scooter_1': 250, 'e_car': 5000}
    energy_consumed = vehicle_efficiency[current_mode] * distance
    # Calculate the delta SoC (%) for the distance traveled
    delta_soc = (energy_consumed / battery_capacity[current_mode]) * 100

    return delta_soc

