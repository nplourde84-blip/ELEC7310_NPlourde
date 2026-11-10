#main 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
#from mpl_toolkits.mplot3d import Axes3D
EPS0 = 8.854e-12   # F/m — permittivity of free space
MU0  = 4*np.pi*1e-7  # H/m — permeability of free space
C    = 1/np.sqrt(MU0 * EPS0)  # m/s — speed of light
f0 = 20*(10**6) # 20 MHz
lambda0 = C/f0  # m — wavelength
epsR = 1.0 # Relative permittivity of the medium (air)

xmax = 1000 # Maximum length of the domain in meters
dx = 0.01 # Grid spacing in meters

S = 0.5 # Courant number for stability
dt = S * dx * np.sqrt(epsR) / C # Time step in seconds

t = 0 # Initialize time
tmax = 1e-6 # Maximum simulation time in seconds

E = np.zeros((int(xmax/dx), 3)) # Initialize electric field array
n = 0 # Initialize time step counter
while n < tmax: # Run the simulation for the specified time
    # Update the electric and magnetic fields using FDTD equations
    E(i,n+1) = 1/(MU0*EPS0*epsR(i))*(dt/dx)^2 * [E(i+1,n) - 2*E(i,n) + E(i-1,n)] + 2*E(i,n) - E(i,n-1)
