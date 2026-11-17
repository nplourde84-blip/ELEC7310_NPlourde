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
epsRA = 2.0 # Relative permittivity of the medium
epsRB = 3.0 # Relative permittivity of the medium
xmax = 100 # Maximum length of the domain in meters
dx = 0.01 # Grid spacing in meters

S = 0.5 # Courant number for stability
dt = S * dx * np.sqrt(epsR) / C # Time step in seconds

t = 0 # Initialize time
tmax = 1e-6 # Maximum simulation time in seconds
tau = 5e-9 # Pulse width in seconds
t0 = 5*tau # Time delay for the pulse
def gaussian(t):
    return np.exp(-((t-t0)/tau)**2) # Gaussian pulse centered at t=t0 with width tau
N = int(xmax/dx) # Number of grid points
x = np.arange(N)*dx # Position of each grid point in meters
E = np.zeros((N, 3)) # Initialize electric field array, columns: [n-1, n, n+1]

# Relative permittivity at each grid point
eps = np.full(N, epsR) # air everywhere
eps[(x >= 35) & (x < 40)] = epsRA # Region A: 35 m to 40 m
eps[(x >= 40) & (x < 50)] = epsRB # Region B: 40 m to 50 m
n = 0 # Initialize time step counter
while t < tmax: # Run the simulation for the specified time
    # Update the electric and magnetic fields using FDTD equations
    E[1:-1,2] = 1/(MU0*EPS0*eps[1:-1])*(dt/dx)**2 * (E[2:,1] - 2*E[1:-1,1] + E[:-2,1]) + 2*E[1:-1,1] - E[1:-1,0]
    E[0,2] = gaussian(t + dt) # Apply the Gaussian pulse at the left boundary
    # Right boundary E[-1] stays 0 (perfect conductor, reflects the pulse)

    # Shift time levels for the whole array (so the source value reaches the interior)
    E[:,0] = E[:,1]
    E[:,1] = E[:,2]
    t += dt
    n += 1
