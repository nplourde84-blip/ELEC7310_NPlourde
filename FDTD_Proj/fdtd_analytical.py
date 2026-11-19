import numpy as np

EPS0 = 8.854e-12   # F/m — permittivity of free space
MU0  = 4*np.pi*1e-7  # H/m — permeability of free space
C    = 1/np.sqrt(MU0 * EPS0)  # m/s — speed of light
f0 = 20*(10**6) # 20 MHz
lambda0 = C/f0  # m — wavelength
EPSR = 1.0 # Relative permittivity of air (background)
EPSRA = 2.0 # Relative permittivity of the sides
EPSRB = 3.0 # Relative permittivity of the core

ns = np.sqrt(EPSRA) # Refractive index of sides
nc = np.sqrt(EPSRB) # Refractive index of core

def reflection_coefficient(ns, nc, phi):
    """Calculate the reflection coefficient at a boundary between two media."""
    X = np.sin(phi) * ((ns - 1/ns)*np.cos(phi) + (nc - 1/nc)*np.cos(phi)**2 + (nc/(ns**2) - (ns**2)/nc)*np.sin(phi/2)**2)
    return np.abs(X)/np.sqrt(X**2 + 4)  # Magnitude of the reflection coefficient

if __name__ == "__main__":
    for f in [0, f0/2, f0, 3*f0/2, 2*f0]:  # Test at fundamental and harmonics
        phi = np.pi*f/f0 # Normal incidence
        R = reflection_coefficient(ns, nc, phi)
        print(f"f = {f/f0:.1f} f0 ({f/1e6:.0f} MHz): |R| = {R:.4f}")