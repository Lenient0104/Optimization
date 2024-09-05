# Energy Comsumption Model

In our study, we adopted the mathematical energy consumption model initially proposed in [1], customising it to align with real-world application scenarios specific to our research. According to [2]-[4], it is recognised that motor power is mainly affected by road slope, friction, and air resistance, and we can ascertain some physical parameters. We use $m$, $s$, and $v$ to represent the mass of the device and user, the road slope, and the driving speed respectively. The acceleration of gravity $g$ is approximately 9.81 $\mathrm{m/s^2}$. For asphalt surfaces, the rolling resistance coefficient $C_r$ is typically 0.001. The air density $\rho$ is given as 1.29 $\mathrm{kg/m^3}$. Additionally, the combined frontal area $A$ of the device and rider is 0.5 $\mathrm{m^2}$, and the drag coefficient $C_d$ is 0.7. To calculate the energy consumption per unit distance, we further refined the model and arrived at equation \eqref{eq: math definition}, which serves as the total energy required for \ac{escooter}, i.e., $P_{es} = P_{d}$. For E-bike, we introduce the power assistance level $P_{al}$, which ranges from 0 (no assistance) to 1 (full assistance), and formulate our prediction target, i.e., energy consumption demand per unit distance, as $P_{eb} = P_{d} \cdot P_{al}$.

$$
P_{d} = g \cdot m \cdot s + C_r \cdot m \cdot g + \frac{1}{2} C_d \cdot \rho \cdot A \cdot v^2
$$

For more details, please refer to another paper of ours [5].


[1] E. Burani, G. Cabri, and M. Leoncini, “An algorithm to predict e-bike power consumption based on planned routes,” Electronics, vol. 11, no. 7, p. 1105, Mar. 2022.

[2] W. J. v. Steyn and J. Warnich, “Comparison of tyre rolling resistance for different mountain bike tyre diameters and surface conditions,” South African Journal for Research in Sport, Physical Education and Recreation, vol. 36, no. 2, pp. 179–193, 2014.

[3] B. Upadhya, R. Altoumaimi, and T. Altoumaimi, “Characteristics and control of the motor system in e-bikes,” 2014.

[4] W. M. Bertucci, S. Rogier, and R. F. Reiser, “Evaluation of aerodynamic and rolling resistances in mountain-bike field conditions,” Journal of sports sciences, vol. 31, no. 14, pp. 1606–1613, 2013.

[5] https://arxiv.org/abs/2403.17632

