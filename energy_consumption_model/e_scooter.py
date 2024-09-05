class Escooter_PowerConsumptionCalculator:
    def __init__(self, Cr=0.001, Cd=0.7, A=2.5, rho=1.29, g=9.81):
        self.Cr = Cr  # Rolling resistance coefficient
        self.Cd = Cd  # Resistance coefficient
        self.A = A  # Frontal area
        self.rho = rho  # Air density
        self.g = g  # Gravitational acceleration

    def calculate(self, v, m):
        """
        v -- (Km/s)
        m -- (kg)
        s -- slope
        """
        return (self.Cr * m * self.g + 0.5 * self.Cd * self.rho * self.A * v * v)
