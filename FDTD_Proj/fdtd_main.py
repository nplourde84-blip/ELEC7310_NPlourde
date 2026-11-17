#main 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation
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

S = 1 # Courant number for stability
dt = S * dx * np.sqrt(epsR) / C # Time step in seconds

t = 0 # Initialize time
tmax = 0.4e-6 # Maximum simulation time in seconds
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
save_every = 200 # Save a snapshot of E every this many time steps
snapshots = [] # Saved E-field snapshots for plotting
snap_times = [] # Time of each snapshot in seconds
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

    if n % save_every == 0:
        snapshots.append(E[:,1].copy())
        snap_times.append(t)

snapshots = np.array(snapshots)
snap_times = np.array(snap_times)

# ---------------- Plotting ----------------
# Find the epsilon blocks (continuous runs of the same eps value that are not air)
edges = np.flatnonzero(np.diff(eps)) + 1 # Indices where eps changes value
starts = np.concatenate(([0], edges))
ends = np.concatenate((edges, [N]))
blocks = [(x[i0], x[i1-1] + dx, eps[i0]) for i0, i1 in zip(starts, ends) if eps[i0] != epsR]

ylim = 1.1*max(1.0, np.abs(snapshots).max())

def draw_blocks(ax):
    # Shade each epsilon block, draw lines at its edges, and label its eps value
    for xa, xb, e in blocks:
        ax.add_patch(Rectangle((xa, -ylim), xb - xa, 2*ylim, color='0.5', alpha=0.08 + 0.06*e, lw=0))
        ax.axvline(xa, color='0.35', lw=1, ls='--')
        ax.axvline(xb, color='0.35', lw=1, ls='--')
        ax.text((xa + xb)/2, 0.93*ylim, f'$\\varepsilon_r$={e:g}', ha='center', va='top', fontsize=9, color='0.25')

def style_axes(ax):
    ax.set_xlim(0, xmax)
    ax.set_ylim(-ylim, ylim)
    ax.axhline(0, color='0.8', lw=0.8)
    ax.grid(alpha=0.3)

# Static figure: the wave packet at a few times
n_panels = 4
idx = np.linspace(0, len(snapshots) - 1, n_panels + 2).astype(int)[1:-1]
fig1, axs = plt.subplots(n_panels, 1, figsize=(10, 2.2*n_panels), sharex=True)
for ax, i in zip(axs, idx):
    draw_blocks(ax)
    ax.plot(x, snapshots[i], color='C0', lw=2)
    style_axes(ax)
    ax.set_ylabel('E (V/m)')
    ax.set_title(f't = {snap_times[i]*1e9:.0f} ns', loc='left', fontsize=10)
axs[-1].set_xlabel('x (m)')
fig1.tight_layout()
fig1.savefig('fdtd_snapshots.png', dpi=150)

# Animation: the wave packet moving through the epsilon blocks
fig2, ax2 = plt.subplots(figsize=(10, 4))
draw_blocks(ax2)
style_axes(ax2)
ax2.set_xlabel('x (m)')
ax2.set_ylabel('E (V/m)')
line, = ax2.plot(x, snapshots[0], color='C0', lw=2)
title = ax2.set_title('', loc='left')

def update(k):
    line.set_ydata(snapshots[k])
    title.set_text(f't = {snap_times[k]*1e9:.0f} ns')
    return line, title

anim = FuncAnimation(fig2, update, frames=len(snapshots), interval=30, blit=False)
plt.show()
