import Millow as Millow

def sizeTest():
    for Testsize in [(1234,4564), (1000,1000), (45,900)]:
        map = Millow.Millow('sparse islands', mapSize=Testsize)
        map.toImage()
        assert map.img.size == Testsize[::-1], 'Size should be {}, but is {}'.format(Testsize, map.img.size)

if __name__=='__main__':
    sizeTest()
    print('Passed all tests.')