import numpy as np

from Mesh import Mesh

# Define mesh and material for the shaft
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
        [6, 6, 6, 6],
    ]
)

msh = Mesh(shaft_dim)
msh.set_density(7800)
msh.set_emod(2.0e11)

print(msh.elements)
