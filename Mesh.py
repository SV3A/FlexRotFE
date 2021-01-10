import numpy as np


class Mesh:

    """
    Object representing discretizised shaft elements with associated material
    properties.
    """

    def __init__(self, mesh: np.ndarray):

        # Total number of elements
        self.num_el = np.sum(mesh[3, :])

        # Discretization matrix
        self.elements = np.zeros((5, self.num_el))

        self._set_geometry(mesh)

    def _set_geometry(self, mesh):
        """
        Define the geomtry of the shaft, i.e. length, outer radius, and inner
        radius of each element.
        """

        start_idx = 0

        for ii, n_els in enumerate(mesh[3, :]):

            end_idx = start_idx + n_els

            # Properties for each element of the current segment
            el_length = 1e-3 * mesh[0, ii] / n_els
            el_out_radius = 1e-3 * mesh[1, ii]
            el_in_radius = 1e-3 * mesh[2, ii]

            # Insert into the discretization matrix
            self.elements[0, start_idx:end_idx] = np.repeat(el_length, n_els)
            self.elements[1, start_idx:end_idx] = np.repeat(el_out_radius, n_els)
            self.elements[2, start_idx:end_idx] = np.repeat(el_in_radius, n_els)

            start_idx = start_idx + n_els

    def set_density(self, rho: float, *args, **kwargs):
        """
        Set the desity an individual element or all the elements at once.
        """

        # Optional argument specifying which element to assign density to
        el = kwargs.get("element", None)

        if el:
            self.elements[3, el-1] = rho
        else:
            self.elements[3, :] = rho

    def set_emod(self, e_module: float, *args, **kwargs):
        """
        Set Young's module of an individual element or all the elements at once.
        """

        # Optional argument specifying which element to assign density to
        el = kwargs.get("element", None)

        if el:
            self.elements[4, el-1] = e_module
        else:
            self.elements[4, :] = e_module
