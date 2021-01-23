import numpy as np

from Mesh import Mesh
from RotorFEModel import RotorFEModel, Disc, Bearing

# Define mesh (discretization) for the shaft
# Rows:
#   0: Lenght [mm]
#   1: Outer radius [mm]
#   2: Inner radius [mm]
#   3: Partition number of the element []
shaft_dim = np.array(
    [
        [200, 200, 200, 200],
        [10, 20, 30, 10],
        [0, 0, 0, 0],
        [6, 6, 6, 6]
    ]
)

msh = Mesh(shaft_dim)

# Set shaft material
msh.set_density(7800)
msh.set_emod(2.0e11)


# Init FE model
rot_mod = RotorFEModel(msh.elements)
rot_mod.add_prop_damping(0, 5e-05)

# Define machine elements and add them to the global system
disc_1 = Disc(20, 1.5e-1, 1.5e-1, u=3e-5)

bearing_1 = Bearing(np.array([
    [10e2,  0],
    [0,  10e2]
]))

rot_mod.add_node_component(1, bearing_1)
rot_mod.add_node_component(25, bearing_1)
rot_mod.add_node_component(13, disc_1)

rot_mod.print_info()

# Write M, G, K, and D matrices to disk
# rot_mod.export("./exports/mod1")
