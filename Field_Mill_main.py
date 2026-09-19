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
# 2. Create a field mill with a given geometry and parameters
# 3. Define field disribution in the domain space
# 4. Make Gauss' law calculations to determine the field at the field mill
# 5. Calculate the field at the mill and show the plot

print(f"c = 1/√(μ₀ε₀) = {C:.6e} m/s")

def plot_electric_field():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    x = np.linspace(-2, 2, 24)
    y = np.linspace(-2, 2, 24)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    R = np.where(R < 0.15, 0.15, R)  # avoid singularity at origin
    
    # Coulomb's law: E = kq/r² in the radial direction
    Ex = X / R**3
    Ey = Y / R**3
    E_mag = np.sqrt(Ex**2 + Ey**2)
    
    # Positive charge
    ax = axes[0]
    ax.streamplot(X, Y, Ex, Ey, color=np.log(E_mag+1), cmap='Reds', density=1.5)
    ax.add_patch(Circle((0,0), 0.18, color='crimson', zorder=4))
    ax.text(0, 0, '+q', ha='center', va='center', fontsize=10,
            color='white', fontweight='bold', zorder=5)
    ax.set_aspect('equal')
    ax.set_title('E field: Positive Charge\n∇·E = ρ/ε₀ — field diverges outward')
    ax.grid(True, alpha=0.25)
    
    # Negative charge
    ax = axes[1]
    ax.streamplot(X, Y, -Ex, -Ey, color=np.log(E_mag+1), cmap='Blues', density=1.5)
    ax.add_patch(Circle((0,0), 0.18, color='steelblue', zorder=4))
    ax.text(0, 0, '−q', ha='center', va='center', fontsize=10,
            color='white', fontweight='bold', zorder=5)
    ax.set_aspect('equal')
    ax.set_title('E field: Negative Charge\nfield converges inward')
    ax.grid(True, alpha=0.25)
    
    plt.tight_layout()
    plt.savefig('electric_field.png', dpi=150)
    plt.show()

def plot_magnetic_field():
    fig, ax = plt.subplots(figsize=(7, 7))
    
    x = np.linspace(-2, 2, 22)
    y = np.linspace(-2, 2, 22)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    R = np.where(R < 0.15, 0.15, R)
    
    # Right-hand rule: current out of page (+z), B is circumferential
    # Bx = -y/r², By = x/r²
    Bx = -Y / R**2
    By =  X / R**2
    B_mag = np.sqrt(Bx**2 + By**2)
    
    ax.streamplot(X, Y, Bx, By, color=np.log(B_mag+1), cmap='Greens', density=1.5)
    
    # Wire symbol: dot = current coming toward you
    ax.add_patch(Circle((0,0), 0.18, color='crimson', zorder=4))
    ax.plot(0, 0, 'w.', ms=7, zorder=5)
    ax.text(0.25, 0.25, 'I (out of page)', color='crimson', fontsize=10)
    
    ax.set_aspect('equal')
    ax.set_title('B field: Current-Carrying Wire\n∇×B = μ₀J — field circulates around current')
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig('magnetic_field.png', dpi=150)
    plt.show()

def plot_poynting_coaxial():
    a = 0.3   # inner radius
    b = 1.0   # outer radius
    V = 50.0  # voltage [V]
    I = 2.0   # current [A]
    
    # Field profiles vs radius
    r = np.linspace(a + 0.01, b - 0.01, 300)
    E_r   = V / (r * np.log(b/a))          # V/m
    H_phi = I / (2 * np.pi * r)            # A/m
    S_z   = E_r * H_phi                    # W/m²
    
    # Integrate to get total power
    P = np.trapezoid(S_z * 2 * np.pi * r, r)
    print(f"V = {V} V,  I = {I} A,  VI = {V*I} W")
    print(f"∫ S_z · 2πr dr = {P:.4f} W")
    # These two numbers will be equal.
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Radial profiles
    ax = axes[0]
    ax2 = ax.twinx()
    ax.plot(r, E_r,   'crimson',   lw=2, label='E_r (V/m)')
    ax.plot(r, H_phi, 'steelblue', lw=2, label='H_φ (A/m)')
    ax2.plot(r, S_z,  'purple',    lw=2.5, ls='--', label='S_z (W/m²)')
    ax.set_xlabel('Radius r [m]')
    ax.set_ylabel('Field strength')
    ax2.set_ylabel('Poynting flux [W/m²]', color='purple')
    ax.set_title(f'Fields in Dielectric\n∫S_z·2πr dr = {P:.2f} W = VI ✓')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Cross-section heat map
    ax = axes[1]
    theta = np.linspace(0, 2*np.pi, 360)
    r_2d, T_2d = np.meshgrid(np.linspace(a, b, 200), theta)
    S_map = (V * I) / (2 * np.pi * r_2d**2 * np.log(b/a))
    
    pc = ax.pcolormesh(r_2d*np.cos(T_2d), r_2d*np.sin(T_2d),
                       S_map, cmap='plasma', shading='auto')
    plt.colorbar(pc, ax=ax, label='S_z [W/m²]')
    ax.add_patch(Circle((0,0), a, color='orange', label='Inner conductor'))
    ax.add_patch(Circle((0,0), b, color='gray', fill=False, lw=3, label='Outer conductor'))
    ax.set_aspect('equal')
    ax.set_title('Poynting Vector in Cross-Section\nEnergy flows through the INSULATOR')
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig('poynting_coaxial.png', dpi=150)
    plt.show()

def drift_velocity_table():
    rho_Cu = 8960.0
    M_Cu   = 63.546e-3
    N_A    = 6.022e23
    e      = 1.602e-19
    n      = rho_Cu * N_A / M_Cu
    
    print(f"Copper free electron density: n = {n:.3e} /m³\n")
    
    cases = [
        ("House wiring (1 mm², 10 A)",    1e-6,   10.0),
        ("House wiring (1 mm²,  1 A)",    1e-6,    1.0),
        ("USB-C cable (0.5 mm², 3 A)",    0.5e-6,  3.0),
        ("EV fast charger (50 mm², 200A)", 50e-6, 200.0),
    ]
    
    snail = 13.9e-3  # m/s
    
    for name, A, I in cases:
        v_d = I / (n * A * e)
        ratio = snail / v_d
        print(f"{name}")
        print(f"  v_d = {v_d*1e3:.4f} mm/s  |  snail is {ratio:.0f}x faster\n")
    
    # Time to travel 100 km
    v_typical = 10 / (n * 1e-6 * e)
    years = 1e5 / v_typical / (365.25 * 86400)
    print(f"Time for electron to travel 100 km: {years:.0f} years")

def plot_em_wave():
    fig = plt.figure(figsize=(13, 6))
    ax  = fig.add_subplot(111, projection='3d')
    
    lam = 1.0
    k   = 2*np.pi/lam
    z   = np.linspace(0, 3*lam, 400)
    
    Ex = np.sin(k*z)  # electric field, x-direction
    By = np.sin(k*z)  # magnetic field, y-direction
    # They're in phase, perpendicular to each other,
    # and both perpendicular to the propagation direction z.
    # S = E × B/μ₀ points in +z. Energy propagates in the direction of travel.
    
    ax.plot(Ex, np.zeros_like(z), z, 'r-', lw=2, label='E field')
    ax.plot(np.zeros_like(z), By, z, 'b-', lw=2, label='B field')
    
    # Arrows at intervals
    for zi in np.linspace(0.1, 3*lam-0.1, 18):
        ei = np.sin(k*zi)
        if abs(ei) > 0.1:
            ax.quiver(0, 0, zi, ei*0.7, 0, 0, color='crimson', alpha=0.65,
                      arrow_length_ratio=0.35)
            ax.quiver(0, 0, zi, 0, ei*0.7, 0, color='steelblue', alpha=0.65,
                      arrow_length_ratio=0.35)
    
    ax.set_xlabel('E direction'); ax.set_ylabel('B direction')
    ax.set_zlabel('Propagation z')
    ax.set_title(f'Electromagnetic Wave\nE ⊥ B ⊥ ẑ  |  c = {C:.3e} m/s')
    ax.legend()
    ax.view_init(elev=22, azim=-58)
    plt.tight_layout()
    plt.savefig('em_wave.png', dpi=150)
    plt.show()

if __name__ == "__main__":
    plot_electric_field()
    plot_magnetic_field()
    plot_poynting_coaxial()
    drift_velocity_table()
    plot_em_wave()
