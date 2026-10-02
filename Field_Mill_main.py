#main
#This code is to create an environment to show the use of Gauss' law in a field mill. 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import Axes3D
EPS0 = 8.854e-12   # F/m — permittivity of free space
MU0  = 4*np.pi*1e-7  # H/m — permeability of free space
C    = 1/np.sqrt(MU0 * EPS0)  # m/s — speed of light
# 1. Create Domain space
    # We can assume that the domain space is 2D if the field mill contains an infinite y component.
class Domain:
    def __init__(self, xlim, ylim):
        self.xlim = xlim
        self.ylim = ylim
        self.X, self.Y = np.meshgrid(np.linspace(xlim[0], xlim[1], 100), np.linspace(ylim[0], ylim[1], 100))
        self.Z = np.zeros_like(self.X)  # Initialize Z to zero for 2D field distribution

# 2. Create a field mill with a given geometry and parameters
    # Create a class to represent the field mill, including its geometry and parameters.

# 3. Define field disribution in the domain space
# 4. Make Gauss' law calculations to determine the field at the field mill
# 5. Calculate the field at the mill and show the plot

if __name__ == "__main__":
    # Define the domain space
    domain = Domain(xlim=(-5, 5), ylim=(-5, 5))
    
    # Create a figure and axis for plotting
    fig, ax = plt.subplots()
    
    # Plot the domain space
    ax.set_xlim(domain.xlim)
    ax.set_ylim(domain.ylim)
    ax.set_title("Field Mill Environment")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    
    # Show the plot
    plt.grid()
    plt.show()