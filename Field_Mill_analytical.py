# This takes the calculations from the analytical model and plots the results. 
# It is a separate file from Field_Mill_main.py to avoid cluttering the main code with plotting and analysis.

import numpy as np
import matplotlib.pyplot as plt
from Field_Mill_main import FieldMill, EPS0

E0 = 100     # V/m — uniform field normal to the plates
R = 1e3      # 1 kOhm - Arbitrary feedback resistance
VEL = 1      # 1 m/s - Arbitrary velocity of the ground plate

def exposed_area(plate, ground, depth):
    covered = np.clip(min(plate.x0 + plate.width, ground.x0 + ground.width) - max(plate.x0, ground.x0),
                      0, plate.width)
    return (plate.width - covered) * depth

if __name__ == "__main__":
    mill = FieldMill(center=(5,5))

    travel = mill.side - mill.ground_plate.width
    offsets = np.linspace(0, travel, 2001)
    t = offsets / VEL

    charges = {plate.name: [] for plate in mill.sense_plates}
    for offset in offsets:
        mill.set_shutter(offset / travel)
        for plate in mill.sense_plates:
            charges[plate.name].append(EPS0 * E0 * exposed_area(plate, mill.ground_plate, mill.depth))
    charges = {name: np.array(q) for name, q in charges.items()}

    I1, I2 = (np.gradient(q,t) for q in charges.values())
    V_out = I1 * R/2 - I2 * R/2
    V_ideal = EPS0 * E0 * mill.depth * VEL * R

    fig, (ax1, ax2) = plt.subplots(2,1, sharex=True)
    for name, q in charges.items():
        ax1.plot(offsets, q*1e9, label=name)
    ax1.set_ylabel("Plate Charge (nC)")
    ax1.set_title("Analytical Field Mill")
    ax1.legend()
    ax1.grid()

    ax2.plot(offsets, V_out*1e6, label="Voltage out")
    ax2.set_ylabel("Output Voltage (V)")
    ax2.set_xlabel("Offset (m)")
    ax2.legend()
    ax2.grid()
    fig.savefig("analytical_soln.png", dpi=200, bbox_inches="tight")
    plt.show()