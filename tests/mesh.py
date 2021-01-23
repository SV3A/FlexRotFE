import sys
import copy
import unittest

sys.path.append("../src/")

import numpy as np
from Mesh import Mesh


class TestMesh(unittest.TestCase):
    def test_dimensions(self):

        # Even number of elements
        shaft_dim = np.array(
            [
                [100, 100],
                [10, 20],
                [0, 5],
                [2, 2],
            ]
        )

        msh = Mesh(shaft_dim)

        self.assertEqual(msh.elements.size, 20, "number of elements are wrong")

        self.assertEqual(msh.elements.shape, (5, 4), "matrix shape is wrong")

        # Odd number of elements
        shaft_dim = np.array(
            [
                [100, 100],
                [10, 20],
                [0, 5],
                [3, 2],
            ]
        )

        msh = Mesh(shaft_dim)

        self.assertEqual(msh.elements.size, 25, "number of elements are wrong")

        self.assertEqual(msh.elements.shape, (5, 5), "matrix shape is wrong")

    def test_material_assignment(self):

        # Row assignment (density)
        shaft_dim = np.array(
            [
                [100, 100],
                [10, 20],
                [0, 5],
                [2, 2],
            ]
        )

        msh = Mesh(shaft_dim)

        old_mesh = copy.deepcopy(msh.elements)

        msh.set_density(20)

        expected = np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [20.0, 20.0, 20.0, 20.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )

        np.testing.assert_array_equal((msh.elements - old_mesh), expected)


        # Index assignment (density)
        old_mesh = copy.deepcopy(msh.elements)

        msh.set_density(30, element=1)
        msh.set_density(40, element=3)

        expected = np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [10.0, 0.0, 20.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )

        np.testing.assert_array_equal((msh.elements - old_mesh), expected)

        # Row assignment (e-module)
        old_mesh = copy.deepcopy(msh.elements)
        msh.set_emod(50)

        expected = np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [50.0, 50.0, 50.0, 50.0],
            ]
        )

        np.testing.assert_array_equal((msh.elements - old_mesh), expected)

        # Index assignment (e-module)
        old_mesh = copy.deepcopy(msh.elements)

        msh.set_emod(60, element=2)
        msh.set_emod(70, element=4)

        expected = np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 10.0, 0.0, 20.0],
            ]
        )

        np.testing.assert_array_equal((msh.elements - old_mesh), expected)

if __name__ == "__main__":
    unittest.main()
