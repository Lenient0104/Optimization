import random
import numpy as np


class ECar_EnergyConsumptionModel:
    def __init__(self, numb):
        # aerodynamic parameter
        self.Paer = np.random.normal(3.12e-5)
        # parameter for rolling resistance with different passengers
        self.Ptir = 0.05 / 1235 * (1235 + numb * 80)
        # random power loading for ancillary services [0.5, 1.5] (更窄的范围避免极端)
        self.Panc = 0.5 + random.random() * 1
        if self.Paer < 0:
            self.Paer = 0
        # coefficients for the energy loss polynomial
        self.Pa = self.Paer + 4e-6  # x^2 term coefficient
        self.Pb = 5e-4  # x term coefficient
        self.Pc = 0.0293 + self.Ptir  # constant term
        self.Pd = 0.375 + self.Panc  # x^(-1) term coefficient

    def calculate_energy_loss(self, speed):
        # Calculate energy loss
        energy_loss = (self.Pa * speed ** 2 + self.Pb * speed + self.Pc + self.Pd / speed) * 1000 / speed

        # 防止能量损失为负值，最小设定为 0
        return max(energy_loss, 0)


# Example usage
num_passengers = 4
vehicle = ECar_EnergyConsumptionModel(num_passengers)

# Speed of the vehicle in km/h
vehicle_speed = 100

# Calculate energy loss
energy_loss = vehicle.calculate_energy_loss(vehicle_speed)
print(f"Energy loss: {energy_loss} Wh/km")
