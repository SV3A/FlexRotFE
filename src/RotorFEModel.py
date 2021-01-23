import copy
import warnings
import numpy as np
from pathlib import Path


class NodeElement:
    """
    Super class defining nodal elements (machine elements)
    """
    def __init__(self):
        self.nodal_position = None


class Disc(NodeElement):
    """
    Defines a machine element with linear- and rotational inertia as well as
    optional particle unbalance.
    """

    def __init__(
        self,
        mass: float,
        moment_lin: float,
        moment_pol: float,
        u: float = None,
        m0: float = None,
        e: float = None,
    ):
        super().__init__()

        self.m = mass
        self.I_d = moment_lin
        self.I_p = moment_pol

        # Optionally accept unbalance
        self._eval_unbalance_args(u, m0, e)

        self.e = e
        self.m0 = m0

        if m0 and e:
            self.u = m0 * e
        else:
            self.u = u

        if self.u is None:
            self.has_unbalance = False
        else:
            self.has_unbalance = True

    def _eval_unbalance_args(self, u: float, m0: float, e: float) -> None:
        """
        Ensure that 1) only "u" was supplied, or 2) "m0" and "e" was supplied,
        if not raise ValueError.
        """

        rejections = [
            ((u and m0) or (u and e)) is not None,
            m0 is not None and e is None,
            e is not None and m0 is None,
        ]

        if True in rejections:
            raise ValueError(
                "Specify unbalance either with a single unbalance parameter "
                + "'u' or with discrete unbalance mass 'm0' and eccentricity 'e'"
            )


class Bearing(NodeElement):
    """
    Defines a machine element with linear- and/or rotational stiffness
    """

    def __init__(self, local_K: np.ndarray):
        super().__init__()

        self.local_K = local_K


class RotorFEModel:
    """
    Implements the finite element method on a rotor.  In addition to the shaft,
    nodal elements can be mounted onto the rotor, e.g. bearings or discs.
    """

    def __init__(self, elements: np.ndarray):

        # Counts of the FE model
        self.n_elements = np.size(elements, 1)
        self.n_nodes = self.n_elements + 1
        self.n_dofs = self.n_nodes * 4

        # Set size of system matrices
        self.M = np.zeros((self.n_dofs, self.n_dofs))
        self.G = np.zeros((self.n_dofs, self.n_dofs))
        self.K = np.zeros((self.n_dofs, self.n_dofs))
        self.D = np.zeros((self.n_dofs, self.n_dofs))
        self.damped = False

        self.node_elems = []

        self.build_shaft_matrices(elements)

    def build_shaft_matrices(self, elements: np.ndarray) -> None:
        """
        Builds the global matrices M, K, and G.
        """

        # Start- and end indices
        a = 0
        b = 7

        MI_FACTOR = np.pi * 0.25  # pi/4

        for e in range(self.n_elements):

            # Element properties
            l = elements[0, e]
            ro = elements[1, e]
            ri = elements[2, e]
            rho = elements[3, e]
            e_mod = elements[4, e]

            lsq = l * l
            trans_area = np.pi * (ro ** 2 - ri ** 2)
            mom_inert = MI_FACTOR * (ro ** 4 - ri ** 4)

            # Mass matrices
            # Linear inertia matrix
            local_M_lin = np.array([
                [ 156,   0,     0,      22*l,   54,    0,     0,     -13*l  ],
                [ 0,     156,  -22*l,   0,      0,     54,    13*l,   0     ],
                [ 0,    -22*l,  4*lsq,  0,      0,    -13*l, -3*lsq,  0     ],
                [ 22*l,  0,     0,      4*lsq,  13*l,  0,     0,     -3*lsq ],
                [ 54,    0,     0,      13*l,   156,   0,     0,     -22*l  ],
                [ 0,     54,   -13*l,   0,      0,     156,   22*l,   0     ],
                [ 0,     13*l, -3*lsq,  0,      0,     22*l,  4*lsq,  0     ],
                [-13*l,  0,     0,     -3*lsq, -22*l,  0,     0,      4*lsq ]
            ])

            local_M_lin = local_M_lin * ((rho * trans_area * l) / 420.0)

            # Angular inertia matrix
            local_M_rot = np.array([
                [ 36,   0,    0,      3*l,   -36,   0,    0,      3*l   ],
                [ 0,    36,  -3*l,    0,      0,   -36,  -3*l,    0     ],
                [ 0,   -3*l,  4*lsq,  0,      0,    3*l, -lsq,    0     ],
                [ 3*l,  0,    0,      4*lsq, -3*l,  0,    0,     -lsq   ],
                [-36,   0,    0,     -3*l,    36,   0,    0,     -3*l   ],
                [ 0,   -36,   3*l,    0,      0,    36,   3*l,    0     ],
                [ 0,   -3*l, -lsq,    0,      0,    3*l,  4*lsq,  0     ],
                [ 3*l,  0,    0,     -lsq,   -3*l,  0,    0,      4*lsq ]
            ])

            local_M_rot = local_M_rot * (
                (rho * trans_area * (ro ** 2 - ri ** 2)) / (120.0 * l)
            )

            # Add the inertia matrices
            local_M_lin = local_M_lin + local_M_rot

            # Gyro matrix
            local_G = np.array([
                [ 0,   -36,   3*l,    0,     0,    36,   3*l,    0     ],
                [ 36,   0,    0,      3*l,  -36,   0,    0,      3*l   ],
                [-3*l,  0,    0,     -4*lsq, 3*l,  0,    0,      lsq   ],
                [ 0,   -3*l,  4*lsq,  0,     0,    3*l, -lsq,    0     ],
                [ 0,    36,  -3*l,    0,     0,   -36,  -3*l,    0     ],
                [-36,   0,    0,     -3*l,   36,   0,    0,     -3*l   ],
                [-3*l,  0,    0,      lsq,   3*l,  0,    0,     -4*lsq ],
                [ 0,   -3*l, -lsq,    0,     0,    3*l,  4*lsq,  0     ]
            ])

            local_G = local_G * (
                2.0 * (rho * trans_area * (ro ** 2 + ri ** 2) / (120.0 * l))
            )

            # Stiffness matrix
            local_K = np.array([
                [ 12,   0,    0,      6*l,   -12,   0,    0,      6*l   ],
                [ 0,    12,  -6*l,    0,      0,   -12,  -6*l,    0     ],
                [ 0,   -6*l,  4*lsq,  0,      0,    6*l,  2*lsq,  0     ],
                [ 6*l,  0,    0,      4*lsq, -6*l,  0,    0,      2*lsq ],
                [-12,   0,    0,     -6*l,    12,   0,    0,     -6*l   ],
                [ 0,   -12,   6*l,    0,      0,    12,   6*l,    0     ],
                [ 0,   -6*l,  2*lsq,  0,      0,    6*l,  4*lsq,  0     ],
                [ 6*l,  0,    0,      2*lsq, -6*l,  0,    0,      4*lsq ]
            ])

            local_K = local_K * ((e_mod * mom_inert) / l ** 3)

            # Construct the global mass- and gyro matrix (size n_dofs x n_dofs)
            for ii in range(a, b):
                for jj in range(a, b):
                    self.M[ii, jj] += local_M_lin[ii - e * 4, jj - e * 4]
                    self.G[ii, jj] += local_G[ii - e * 4, jj - e * 4]
                    self.K[ii, jj] += local_K[ii - e * 4, jj - e * 4]

            a += 4
            b += 4

    def add_prop_damping(self, alpha: float, beta: float) -> None:
        """
        Enables proportional (Rayleigh) damping
        """

        self.damped = True

        self.D += alpha * self.M + beta * self.K

    def add_node_component(self, node: int, component: NodeElement) -> None:
        """
        Adds external components (machine elements) to the rotor.
        """
        # Check bounds
        if node > self.n_nodes:
            warnings.warn("Nodal number out of bounds, ignoring component.")
            return

        # Check type of component to be added, and add accordingly
        # Element start index
        es = (node - 1) * 4

        if isinstance(component, Disc):
            self.M[es, es] += component.m
            self.M[es + 1, es + 1] += component.m
            self.M[es + 2, es + 2] += component.I_d
            self.M[es + 3, es + 3] += component.I_d

            self.G[es + 2, es + 3] -= component.I_p
            self.G[es + 3, es + 2] += component.I_p

        elif isinstance(component, Bearing):

            # Handle 2x2 or 4x4 stiffness matrices
            array_size = np.size(component.local_K, 1)

            self.K[es : es+array_size, es : es+array_size] += component.local_K

        # Add component to internal component list (used for info and debug)
        component.nodal_position = node

        # Append component to internal list
        self.node_elems.append(copy.deepcopy(component))

    def print_info(self) -> None:
        """
        Prints properties of the FE model.
        """

        print(f"\nFE-model info:\n  Number of elements: {self.n_elements}")
        print(f"  Number of DOFs: {self.n_dofs}")
        print(f"  Internally damped: {'yes' if self.damped else 'no'}")
        print(f"  Nodal components added: {len(self.node_elems)}")
        for comp in self.node_elems:
            print(
                f"    - {comp.__class__.__name__} at node {comp.nodal_position}"
            )

    def export(self, directory: str):
        """
        Exports the rotor system in terms of the system matrices to a text file.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        # Collect matrices, add only damping matrix if it is non-zero
        matrices = {
            "M": self.M,
            "G": self.G,
            "K": self.K,
        }

        if self.damped:
            matrices["D"] = self.D

        # Write text files
        for label, matrix in matrices.items():
            np.savetxt(directory / f"{label}.txt", matrix)
