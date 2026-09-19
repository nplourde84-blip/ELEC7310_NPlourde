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


if __name__ == "__main__":
    plot_electric_field()
