import Millow
import unittest


class inputTests(unittest.TestCase):
    def test_bad_map_type(self):

        for test_map_type in [50, (50, 1), True, None, 6.5, [1, 2, 3]]:
            self.assertRaises(
                TypeError, Millow.Millow,test_map_type, map_size=(100, 100)
            )

        for test_map_type in ['beans', 'True', '5']:
            self.assertRaises(
                ValueError, Millow.Millow,test_map_type, map_size=(100, 100)
            )


class outputTests(unittest.TestCase):
    def test_size(self):
        for test_size in [(1234, 4564), (1000, 1000), (45, 900)]:
            map = Millow.Millow("sparse islands", map_size=test_size)
            map.generate_basic()
            self.assertEqual(
                map.img.size,
                test_size[::-1],
                "Size should be {}, but is {}".format(test_size, map.img.size),
            )

    def test_size_heights(self):
        for test_size in [(1234, 4564), (1000, 1000), (45, 900)]:
            map = Millow.Millow("sparse islands", map_size=test_size)
            map.generate_basic()
            map.add_height_colouring()
            self.assertEqual(
                map.img.size,
                test_size[::-1],
                "Size should be {}, but is {}".format(test_size, map.img.size),
            )

    def test_size_grid(self):
        for test_size in [(1234, 4564), (1000, 1000), (45, 900)]:
            map = Millow.Millow("sparse islands", map_size=test_size)
            map.generate_basic()
            map.add_grid(grid_size=(10, 10))
            self.assertEqual(
                map.img.size,
                test_size[::-1],
                "Size should be {}, but is {}".format(test_size, map.img.size),
            )

    def test_size_all(self):
        for test_size in [(1234, 4564), (1000, 1000), (45, 900)]:
            map = Millow.Millow("sparse islands", map_size=test_size)
            map.generate_basic()
            map.add_grid(grid_size=(10, 10))
            map.add_height_colouring()
            self.assertEqual(
                map.img.size,
                test_size[::-1],
                "Size should be {}, but is {}".format(test_size, map.img.size),
            )


if __name__ == "__main__":
    unittest.main()