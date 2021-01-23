import sys
import unittest

sys.path.append("../")
from RotorFEModel import RotorFEModel, Disc, Bearing


class TestNodalElements(unittest.TestCase):
    def test_disc_exceptions(self):
        """
        Test that it's only possible to supply either 'u', or
        'm0' togher with 'e'
        """

        msg = "'u' cannot be supplied with 'm0' and 'e'"
        with self.assertRaises(ValueError, msg=msg):
            Disc(1, 1, 1, u=3e-5, m0=1e-3, e=30e-3)

        msg = "'m0' cannot be supplied with 'u'"
        with self.assertRaises(ValueError, msg=msg):
            Disc(1, 1, 1, u=3e-5, m0=1e-3)

        msg = "'e' cannot be supplied with 'u'"
        with self.assertRaises(ValueError, msg=msg):
            Disc(1, 1, 1, u=3e-5, e=30e-3)

        msg = "'m0' cannot be supplied without 'e'"
        with self.assertRaises(ValueError, msg=msg):
            Disc(1, 1, 1, m0=1e-3)

        msg = "'e' cannot be supplied without 'm0'"
        with self.assertRaises(ValueError, msg=msg):
            Disc(1, 1, 1, e=30e-3)

        try:
            Disc(1, 1, 1)
            Disc(1, 1, 1, u=3e-5)
            Disc(1, 1, 1, m0=1e-3, e=30e-3)
        except:
            self.fail("proper assignment raised an exception")

    def test_disc_assignment(self):
        """
        Test the two methods of specifying unbalance
        """
        disc1 = Disc(1.1, 2.1, 3.1)
        self.assertEqual(disc1.m, 1.1, "'m' not assigned correctly")
        self.assertEqual(disc1.I_d, 2.1, "'I_d' not assigned correctly")
        self.assertEqual(disc1.I_p, 3.1, "'I_p' not assigned correctly")

        disc2 = Disc(1.1, 2.1, 3.1, u=3e-5)
        self.assertEqual(disc2.m, 1.1, "'m' not assigned correctly")
        self.assertEqual(disc2.I_d, 2.1, "'I_d' not assigned correctly")
        self.assertEqual(disc2.I_p, 3.1, "'I_p' not assigned correctly")
        self.assertEqual(disc2.u, 3e-5, "'u' not assigned correctly")

        disc3 = Disc(1.1, 2.1, 3.1, m0=1e-3, e=30e-3)
        self.assertEqual(disc3.m, 1.1, "'m' not assigned correctly")
        self.assertEqual(disc3.I_d, 2.1, "'I_d' not assigned correctly")
        self.assertEqual(disc3.I_p, 3.1, "'I_p' not assigned correctly")
        self.assertEqual(disc3.m0, 1e-3, "'m0' not assigned correctly")
        self.assertEqual(disc3.e, 30e-3, "'e' not assigned correctly")
        self.assertEqual(disc3.u, 3e-5, "'u' not assigned correctly")


if __name__ == "__main__":
    unittest.main()
