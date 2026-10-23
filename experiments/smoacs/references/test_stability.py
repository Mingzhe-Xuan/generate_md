import unittest
from assess import stable_status


class StabilityTests(unittest.TestCase):
    def test_incomplete_hull_cannot_certify_stability(self):
        self.assertIsNone(stable_status(.01, False))
        self.assertIsNone(stable_status(None, True))
        self.assertTrue(stable_status(.09999, True))
        self.assertFalse(stable_status(.1, True))
        self.assertFalse(stable_status(.10001, True))
        self.assertFalse(stable_status(.2, False))


if __name__ == '__main__':
    unittest.main()
