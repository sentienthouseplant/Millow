import Millow as Millow


def size_test():
    for Testsize in [(1234, 4564), (1000, 1000), (45, 900)]:
        map = Millow.Millow("sparse islands", mapSize=Testsize)
        map.generate_basic()
        map.add_height()
        map.add_grid((20, 20))
        assert map.img.size == Testsize[::-1], "Size should be {}, but is {}".format(
            Testsize, map.img.size
        )


if __name__ == "__main__":
    size_test()
    print("Passed all tests.")
