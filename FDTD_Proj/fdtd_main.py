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
EPSR = 1.0 # Relative permittivity of air (background)
EPSRA = 2.0 # Relative permittivity of region A
EPSRB = 3.0 # Relative permittivity of region B
xmax = 100 # Maximum length of the domain in meters
dx = 0.01 # Grid spacing in meters

S = 1 # Courant number for stability
dt = S * dx * np.sqrt(EPSR) / C # Time step in seconds

t = 0 # Initialize time
tmax = 0.8e-6 # Maximum simulation time in seconds (long enough for the slab echoes to die out)
tau = 5e-9 # Pulse width in seconds
t0 = 5*tau # Time delay for the pulse
def gaussian(t):
    return np.exp(-((t-t0)/tau)**2) * np.cos(2*np.pi*f0*(t-t0))  # Gaussian pulse centered at t=t0 with width tau
N = int(xmax/dx) # Number of grid points
x = np.arange(N)*dx # Position of each grid point in meters
E = np.zeros((N, 3)) # Initialize electric field array, columns: [n-1, n, n+1]

# Relative permittivity at each grid point
eps = np.full(N, EPSR, dtype=float) # air everywhere (float so values like 2.5 are kept)
eps[(x >= 35) & (x < 40)] = EPSRA # Region A: 35 m to 40 m
eps[(x >= 40) & (x < 50)] = EPSRB # Region B: 40 m to 50 m
x_src = 1.0 # Soft source location in meters
i_src = int(round(x_src/dx))
x_p = 20.0 # Probe location in meters (in air, in front of the slab)
i_p = int(round(x_p/dx))
probe = [] # E at the probe, every time step

n = 0 # Initialize time step counter
save_every = 250 # Save a snapshot of E every this many time steps
snapshots = [] # Saved E-field snapshots for plotting
snap_times = [] # Time of each snapshot in seconds
while t < tmax: # Run the simulation for the specified time
    # Update the electric and magnetic fields using FDTD equations
    E[1:-1,2] = 1/(MU0*EPS0*eps[1:-1])*(dt/dx)**2 * (E[2:,1] - 2*E[1:-1,1] + E[:-2,1]) + 2*E[1:-1,1] - E[1:-1,0]
    # Absorbing (Mur) boundaries at both ends, exact since S=1 in air
    E[0,2] = E[1,1]
    E[-1,2] = E[-2,1]
    # Soft source: inject the change in the pulse so the radiated wave has the shape of gaussian()
    E[i_src,2] += gaussian(t + dt) - gaussian(t)

    # Shift time levels for the whole array (so the source value reaches the interior)
    E[:,0] = E[:,1]
    E[:,1] = E[:,2]
    t += dt
    n += 1
    probe.append(E[i_p,1])

    if n % save_every == 0:
        snapshots.append(E[:,1].copy())
        snap_times.append(t)

snapshots = np.array(snapshots)
snap_times = np.array(snap_times)

# ---------------- Reflection coefficient Gamma(f) ----------------
probe = np.array(probe)
tt = (np.arange(len(probe)) + 1)*dt # Time of each probe sample
# Incident and reflected pulses are well separated in time at the probe, so split the
# signal at the moment the pulse peak reaches the slab front face (x = 35 m)
t_split = t0 + (35 - x_src)/C
inc = np.where(tt < t_split, probe, 0)
refl = np.where(tt >= t_split, probe, 0)

nfft = 2**21 # Zero-pad for a fine frequency grid
f = np.fft.rfftfreq(nfft, dt)
k0 = 2*np.pi*f/C
# Phase shift moves the reference from the probe to the slab front face
Gamma = np.fft.rfft(refl, nfft)/np.fft.rfft(inc, nfft) * np.exp(2j*k0*(35 - x_p))
i0 = np.argmin(np.abs(f - f0))
print(f'Gamma at f0 = {f0/1e6:g} MHz: |Gamma| = {abs(Gamma[i0]):.4f}, '
      f'angle = {np.degrees(np.angle(Gamma[i0])):.2f} deg')

# ---------------- Plotting ----------------
# Find the epsilon blocks (continuous runs of the same eps value that are not air)
edges = np.flatnonzero(np.diff(eps)) + 1 # Indices where eps changes value
starts = np.concatenate(([0], edges))
ends = np.concatenate((edges, [N]))
blocks = [(x[i0], x[i1-1] + dx, eps[i0]) for i0, i1 in zip(starts, ends) if eps[i0] != EPSR]

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
n_panels = 3
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

# Frequency response of the reflection coefficient
band = f <= 100e6
fig3, (ax3a, ax3b) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
ax3a.plot(f[band]/1e6, np.abs(Gamma[band]), color='C1', lw=1.5)
ax3a.set_ylabel('|Γ|')
ax3a.set_title('Reflection coefficient at slab front face (x = 35 m)', loc='left', fontsize=10)
ax3b.plot(f[band]/1e6, np.degrees(np.angle(Gamma[band])), color='C1', lw=1.5)
ax3b.set_ylabel('∠Γ (deg)')
ax3b.set_ylim(-180, 180)
ax3b.set_yticks([-180, -90, 0, 90, 180])
ax3b.set_xlabel('f (MHz)')
ax3b.set_xlim(0, 100)
for ax in (ax3a, ax3b):
    ax.axvline(f0/1e6, color='0.35', lw=1, ls='--')
    ax.grid(alpha=0.3)
ax3a.text(f0/1e6, ax3a.get_ylim()[1], ' f0', va='top', fontsize=9, color='0.25')
fig3.tight_layout()
fig3.savefig('fdtd_gamma.png', dpi=150)

plt.show()
