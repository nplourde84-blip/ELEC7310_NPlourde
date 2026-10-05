#main
#This code is to create an environment to show the use of Gauss' law in a field mill. 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
#from mpl_toolkits.mplot3d import Axes3D
EPS0 = 8.854e-12   # F/m — permittivity of free space
MU0  = 4*np.pi*1e-7  # H/m — permeability of free space
C    = 1/np.sqrt(MU0 * EPS0)  # m/s — speed of light
# 1. Create Domain space
    # We can assume that the domain space is 2D if the field mill contains an infinite y component.
class Domain:
    def __init__(self, xlim, ylim, dx, dy):
        self.xlim = xlim
        self.ylim = ylim
        self.X, self.Y = np.meshgrid(np.linspace(xlim[0], xlim[1], dx), np.linspace(ylim[0], ylim[1], dy))
        self.Z = np.zeros_like(self.X)  # Initialize Z to zero for 2D field distribution

# 2. Create a field mill with a given geometry and parameters
    # Create a class to represent the field mill, including its geometry and parameters.
class Plate:
    def __init__(self, name, x0, y0, width, height, potential=0.0):
        self.name = name
        self.x0 = x0  # lower left corner
        self.y0 = y0
        self.width = width
        self.height = height
        self.potential = potential  # V

    def mask(self, X, Y):
        # True at every grid point that lies inside the plate
        return ((X >= self.x0) & (X <= self.x0 + self.width) &
                (Y >= self.y0) & (Y <= self.y0 + self.height))

class FieldMill:
    def __init__(self, center=(0,0), side=4, gap=1, thick=0.2,
                 n_sense=2, sense_space=0.1, shutter=0):
        self.center = center        # center of the sense plate layer
        self.side = side            # m — side length of the square mill
        self.depth = side           # m — extent into the page (square mill)
        self.gap = gap              # m — distance between sense plates and ground plate
        self.thick = thick          # m — plate thickness in y
        self.n_sense = n_sense
        self.sense_space = sense_space  # m — spacing between neighbouring sense plates
        self.shutter = shutter      # 0 = ground plate at the left edge, 1 = at the right edge  

        self.sense_plates = self._make_sense_plates()
        self.ground_plate = self._make_ground_plate()

    def _make_sense_plates(self):
        cx, cy = self.center
        width = (self.side - (self.n_sense - 1) * self.sense_space) / self.n_sense
        x_left = cx - self.side / 2
        y0 = cy - self.thick / 2
        return [Plate(f"Sense {i + 1}", x_left + i * (width + self.sense_space), y0, width, self.thick)
                for i in range(self.n_sense)]

    def _make_ground_plate(self):
        # The ground plate covers one sense plate at a time and slides in x with the shutter position.
        cx, cy = self.center
        width = self.side / self.n_sense
        x0 = cx - self.side / 2 + self.shutter * (self.side - width)
        y0 = cy + self.thick / 2 + self.gap
        return Plate("Ground", x0, y0, width, self.thick)

    @property
    def plates(self):
        return self.sense_plates + [self.ground_plate]

    def set_shutter(self, shutter):
        self.shutter = shutter
        self.ground_plate = self._make_ground_plate()

    def mask(self, X, Y):
        # True at every grid point that lies inside any conductor
        mask = np.zeros_like(X, dtype=bool)
        for plate in self.plates:
            mask |= plate.mask(X, Y)
        return mask

    def plot(self, ax):
        for plate in self.plates:
            ax.add_patch(Rectangle((plate.x0, plate.y0), plate.width, plate.height,
                    color="tab:orange", label=plate.name))
        ground = self.ground_plate
        ax.add_patch(Rectangle((ground.x0, ground.y0), ground.width, ground.height,
                color="tab:gray", label=ground.name))
            
# 3. Define field disribution in the domain space
class FieldDistribution:
    def __init__(self, domain, field_mill, E0=100):
        self.domain = domain
        self.field_mill = field_mill
        self.field = np.zeros_like(domain.X)  # Initialize field to zero
        self.E0 = E0  # V/m — example constant field
        self.dx = domain.X[0,1] - domain.X[0,0]
        self.dy = domain.Y[1,0] - domain.Y[0,0]

        self.V = np.zeros_like(domain.X)  # V
        self.Ex = np.zeros_like(domain.X)  # V/m
        self.Ey = np.zeros_like(domain.X)  # V/m
        self.E = np.zeros_like(domain.X)  # V/m — magnitude
# 4. Make Gauss' law calculations to determine the field at the field mill
    def solve(self, tol=1e-5, max_iter=100000):
        X,Y = self.domain.X, self.domain.Y
        dx2,dy2 = self.dx**2, self.dy**2

        V = self.E0 * (Y - self.domain.ylim[0])

        fixed = np.zeros_like(X, dtype=bool)
        fixed[0,:] = True
        fixed[-1,:] = True
        for plate in self.field_mill.plates:
            inside = plate.mask(X,Y)
            V[inside] = plate.potential
            fixed |= inside

        #Update Equation
        for i in range(max_iter):
            Vp = np.pad(V, 1, mode="reflect")
            Vp_left = Vp[1:-1, :-2]
            Vp_right = Vp[1:-1, 2:]
            Vp_up = Vp[:-2, 1:-1]
            Vp_down = Vp[2:, 1:-1]
            V_new = ((Vp_left + Vp_right) * dy2)/(2*(dx2 + dy2)) + ((Vp_up +Vp_down) * dx2)/(2*(dx2 + dy2))
            V_new[fixed] = V[fixed]
            change = np.max(np.abs(V_new - V))
            V = V_new

        self.V = V
        self.iterations = i + 1
        dVdy, dVdx = np.gradient(V, self.dy, self.dx)
        self.Ex = -dVdx
        self.Ey = -dVdy
        self.E = np.hypot(self.Ex, self.Ey)
        self.E[self.field_mill.mask(X,Y)] = 0
    
    def plot(self, ax):
        xlim, ylim = self.domain.xlim, self.domain.ylim
        extent = (xlim[0] - self.dx/2, xlim[1] + self.dx/2,
                  ylim[0] - self.dy/2, ylim[1] + self.dy/2)
        image = ax.imshow(self.E, origin="lower", extent=extent, cmap="viridis", vmin=0)
        self.field_mill.plot(ax)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        return image
# 5. Calculate the field at the mill and show the plot

if __name__ == "__main__":
    # Define the domain space
    domain = Domain(xlim=(0, 10), ylim=(0, 10), dx=101, dy=101)
    mill = FieldMill(center=(5.0, 5.0))
    # Create a figure and axis for plotting
    fig, ax = plt.subplots()
    cbar = None
    
    # Move the ground plate one grid spacing at a time,
    # from fully covering the left sense plate to fully covering the right sense plate
    step = domain.X[0, 1] - domain.X[0, 0]          # m — grid spacing in x
    travel = mill.side - mill.ground_plate.width    # m — total distance the ground plate moves
    offsets = np.append(np.arange(0, travel, step), travel)  # last step lands exactly on the right edge

    for offset in offsets:
        mill.set_shutter(offset / travel)
        field = FieldDistribution(domain, mill, E0=100)
        field.solve()
        # Plot the domain space
        ax.clear()
        mill.plot(ax)
        image = field.plot(ax)
        if cbar is None:
            cbar = fig.colorbar(image, ax=ax, label="|E| (V/m)")
        else:
            cbar.update_normal(image)
        ax.set_xlim(domain.xlim)
        ax.set_ylim(domain.ylim)
        ax.set_title(f"Field Mill Environment — ground plate at x = {mill.ground_plate.x0:.2f} m")
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        ax.grid()
        plt.pause(0.05)

    # Show the plot
    plt.show()