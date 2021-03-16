# Outcomes of project.
#
# - Produce a png of a map with various properties stipulated by the user.
# ((159,193,100), (252, 234, 116), (230, 228, 220), (255, 255, 255))

from PIL import Image, ImageDraw
import numpy as np
from scipy.ndimage import zoom, gaussian_filter

# Constants used--------------------------------------------
CRIT_PROB = 0.59274621


# The functions---------------------------------------------

# This function produces a example of percolation. Please see the wikipedia article for more infomation.
# At a special occupation probability we get holes of all sizes (fractals). This would be a good candidate
# for an initial noise array to used for the map.
def noiseArray(sizeTuple):
    prob = CRIT_PROB
    ar = np.random.choice([0, 1], p=[1 - prob, prob], size=sizeTuple)
    return ar


# This function will smooth the resulting noise from the noiseArray. This essentially yields a map with some
# basic landmasses. This is based on the cave generating cellular automata seen in many games.
def cellularSmooth(

    ar: np.array, overPop: int, underPop: int, bornPop: int, steps: int, borderValue=1) -> np.array:

    """Smooths out a noisey array to produce geographical features."""

    ar = np.pad(
        ar, 1, constant_values=borderValue
    )  # Pads the array so we don't have to have special logic for edges.
    height, width = ar.shape  # Gets the width and height of the padded array.

    for step in range(0, steps + 1):

        for i in range(1, width - 1):  # Ignores the padding.
            for j in range(1, height - 1):  # Ignores the padding.

                localNeigh = ar[
                    j - 1 : j + 2, i - 1 : i + 2
                ]  # Numpys indexes are [Inclusive, Exclusive), hence the extra +1.

                land = (
                    localNeigh[1, 1] == 1
                )  # Checks if the center value, the point of the neighbourhood, is land.

                aliveNeighbours = (
                    np.sum(localNeigh) - localNeigh[1, 1]
                )  # Sums the array and subtracts the possible extra value.

                if land:

                    live = (aliveNeighbours >= underPop) and (
                        aliveNeighbours <= overPop
                    )

                    if live:

                        ar[j, i] = 1

                    else:

                        ar[j, i] = 0

                if not land:

                    born = aliveNeighbours > bornPop

                    if born:

                        ar[j, i] = 1

                    else:

                        ar[j, i] = 0

    ar = np.where(ar, 0, 1)  # Switches 0,1 in the array.

    return ar[1:height, 1:width]

# This produces an array of values in the range [-1,1]. Editing the volativity and noiseProb yeilds an assortment of
# possible height arrays.
def heightArray(size: (int, int), volat: float, noiseProb: float):
    """Returns a height profile given a 2-tuple and volatility."""

    if len(size) != 2:
        return "size must be 2D."

    rng = np.random.default_rng()  # A local rng used in this function.

    heightArray = np.zeros(size)  # Initialises the array with zeros.

    heightArray[0, 0] = rng.uniform(
        low=-1, high=1
    )  # The first value is given some initial value from U[0,1].

    heightArray[0, 1] = rng.uniform(low=-1, high=1)

    heightArray[1, 0] = rng.uniform(low=-1, high=1)

    for x0 in range(2, size[1]):  # Iterates over the first row, initialising it.
        if (
            rng.uniform() < noiseProb
        ):  # This is the chance that some independent randomness is introduced.
            heightArray[0, x0] = rng.uniform(low=-1, high=1)
        else:
            uni = rng.uniform(low=-1, high=1)
            point1 = heightArray[0, x0 - 2]  # The value 2 places back in the first row.
            point2 = heightArray[0, x0 - 1]  # The value 1 place back in the first row.
            heightArray[0, x0] = (
                (point1 + point2) / 2
            ) + volat * uni  # Calulates the midpoint and adds a weighted U[-1,1].
        # This should add some dependence that results in sequences of similar values.

    for y0 in range(
        2, size[0]
    ):  # See above for loop. This repeats the above but for the first column.
        if rng.uniform() < noiseProb:
            heightArray[y0, 0] = rng.uniform(low=-1, high=1)
        else:
            uni = rng.uniform(low=-1, high=1)
            point1 = heightArray[y0 - 2, 0]
            point2 = heightArray[y0 - 1, 0]
            heightArray[y0, 0] = ((point1 + point2) / 2) + volat * uni

    for y in range(1, size[0]):  #
        for x in range(1, size[1]):
            if rng.uniform() < noiseProb:
                heightArray[y, x] = rng.uniform(low=-1, high=1)
            else:
                uni = rng.uniform(low=-1, high=1)
                point1 = heightArray[y - 1, x]
                point2 = heightArray[y, x - 1]
                heightArray[y, x] = ((point1 + point2) / 2) + volat * uni

    return heightArray


def colorStrat(alphaArray: np.array, ranges: list, colors: list) -> np.array:
    """Produces a rgba array with colors stratified according to the ranges given."""

    if not (len(colors) == len(ranges)):
        raise Exception("Each color in colors list must have a corresponding range.")

    rArray = gArray = bArray = alphaArray

    rValues = [i[0] for i in colors]

    gValues = [i[1] for i in colors]

    bValues = [i[2] for i in colors]

    (a, b) = alphaArray.shape  # Unpacks size of a, a two dimensional array.

    resultArray = np.full(
        (a, b, 4), [255, 255, 255, 0], dtype=np.uint8
    )  # Creates a rgba array.

    counter = 0

    for low in ranges:
        rArray[(low <= a)] = rValues[counter]
        gArray[(low <= a)] = gValues[counter]
        bArray[(low <= a)] = bValues[counter]

        counter = counter + 1

    resultArray[:, :, 0] = rArray
    resultArray[:, :, 1] = gArray
    resultArray[:, :, 2] = bArray
    resultArray[:, :, 3] = alphaArray

    img = Image.fromarray(resultArray)
    img.show()

    return resultArray


class Millow:
    def __init__(self, mapType: str):
        """Initiates the millow object. 

        Parameters
        ----------
        mapType : A string from the list of possible map types.
            
            Specifies the type of map to generate.

        Raises
        ------
        Exception
            Generates an exception if the given map type is not in the list of map types.
        """

        possTypes = [
            "continents",
            "dense islands",
            "sparse islands",
        ]  # Possible map types, will be updated as I find more.

        if mapType not in possTypes:
            raise Exception(
                "The map type must be one of the possible options."
            )  # Raises an exception if the user
        # attempts to give a invalid choice of map type.

        self.size = (
            1080,
            1920,
        )  # The pixel size of the final image, currently not variable. Will be in future versions.

        self.mapType = mapType  # Sets the map type. Currently unused.

        self.mapTypeDict = {
            "continents": (9, 4, 6, 10, 1),
            "dense islands": (7, 1, 9, 15, 0),
            "sparse islands": (9, 2, 6, 18, 1),
        }
        # A dictionary which maps the users selected map type to the inputs of cellularSmooth. Inputs discovered by
        # experimentation.

        self.rawsGenerated = False  # Used to see if a map has been generated.

    def generateRawMap(self):

        """Generates a basic array with land and water."""

        rawMap = noiseArray(
            (int(self.size[0] / 50), int(self.size[1] / 50))
        )  # Size is divided by 50 since it will be zoomed by this
        # amount to create smooth features. This uses percolation at the critical probability for features of all sizes.

        (a, b, c, d, e) = self.mapTypeDict[
            self.mapType
        ]  # Unpacks the constants used for the smoothing procedure, which
        # produces the desired map type.

        rawMap = cellularSmooth(
            rawMap, a, b, c, d, borderValue=e
        )  # See above for an explanation of a,b,c,d,e. This smooths the noise
        # produced by noiseArray.

        rawMap = zoom(
            rawMap, 50
        )  # Zooming the array restores it to the intended pixel dimensions and smooths out
        # the roughness.

        rawMap = rawMap[
            0:1080, 0:1920
        ]  # Trims any extra entries possibly caused by rounding.

        self.rawMap = rawMap  # Saves the generated smooth map.

    def generateHeight(self):

        """Adds height levels to the rawMap property."""

        alphaArray = noiseArray(
            (int(self.size[0] / 5), int(self.size[1] / 5))
        )  # Size is divided by 50 since it will be zoomed by this
        # amount to create smooth features. This uses percolation at the critical probability for features of all sizes.

        alphaArray = cellularSmooth(
            alphaArray, 9, 2, 6, 17, borderValue=0
        )  # Dense islands.

        alphaArray = zoom(
            alphaArray, 5
        )  # Zooming the array restores it to the intended pixel dimensions and smooths out
        # the roughness.

        alphaArray = alphaArray[
            0:1080, 0:1920
        ]  # Trims any extra entries possibly caused by rounding.

        alphaArray = alphaArray.astype(
            np.float32
        )  # Changes the datatype so that Gaussian blur and divison work with the
        # desired accuracy.

        alphaArray[
            self.rawMap == 0
        ] = 0  # This masks the height array, so only value which correspond to '1's in the
        # rawMap are kept. The rest are set to zero. Thus zero is 'sea level'.

        alphaArray = gaussian_filter(alphaArray, 20)  # Blurs the image significantly.

        alphaArray[
            self.rawMap == 0
        ] = 0  # Remasks the array, now there is some coastal smoothing.

        alphaArray = (
            (alphaArray - np.amin(alphaArray)) / np.amax(alphaArray)
        ) * 255  # All values will be in the range (0,255).

        alphaArray = alphaArray.astype(np.uint8)

        # Gradient Heights

        rArray = np.interp(alphaArray, [0, 200, 255], [159, 252, 253])
        #
        gArray = np.interp(alphaArray, [0, 200, 255], [193, 234, 250])
        #
        bArray = np.interp(alphaArray, [0, 200, 255], [100, 116, 212])

        rawHeight = np.full(
            (1080, 1920, 4), [255, 255, 255, 0], dtype=np.uint8
        )  # Produces a black RGBA array with zero opacity.

        rawHeight[
            :, :, 3
        ] = alphaArray  # Sets the opaticy of the rawHeight array to the one we generated.
        rawHeight[:, :, 0] = rArray
        rawHeight[:, :, 1] = gArray
        rawHeight[:, :, 2] = bArray

        self.rawHeight = rawHeight

    def generateRaws(self):

        "Generates the arrays needed for creating the final image."

        self.generateRawMap()
        self.generateHeight()

        self.rawsGenerated = True

    def toImage(self):

        """Returns a colour pillow image object corresponding to the map."""

        if not self.rawsGenerated:  # Checks if the raw map array has been generated.

            self.generateRaws()

        colourMap = np.zeros(
            [self.size[0], self.size[1], 4], dtype=np.uint8
        )  # Initialises the RGBA array.

        colourMap[self.rawMap == 0] = [79, 76, 176, 255]  # Colours the 'water'.

        colourMap[self.rawMap == 1] = [159, 193, 100, 255]  # Colours the 'earth'.

        baseImg = Image.fromarray(colourMap)

        heightImg = Image.fromarray(self.rawHeight)

        resultImg = Image.alpha_composite(baseImg, heightImg)

        self.img = resultImg

    def addGrid(self,gridDensity: (int,int),lineWidth: int = 2,lineColor = 'white',borderWidth: int = 3):
        """Adds a grid to the map image object. 

        Parameters
        ----------
        gridDensity : A 2-tuple of ints. 
            (Number of squares horizonially, number of squares vertically)
        lineWidth : Int, optional
            The width of the grid lines in pixels, by default 2.
        lineColor : A pillow color, optional
            The color of the lines used for the grid, by default 'white'.
        borderWidth: Int, optional
            The width of the border of the whole map, by default 3.
        """

        draw = ImageDraw.Draw(self.img)

        #Gets the size of the generated image.
        width,top = self.img.size

        #Generates the spacing of the lines.
        spaceWidth = width/gridDensity[0]
        spaceHeight = top/gridDensity[1]

        #Draws the horizonial lines
        for y in range(0,gridDensity[1]):
            
            #Line Coords ((x start, y start), (x end, y end))
            coords = ( (0,y*spaceHeight) , (width,y*spaceHeight))
            draw.line(coords,width=lineWidth,fill=lineColor)

        #Draws the vertical lines
        for x in range(0,gridDensity[0]):
            
            #Line Coords ((x start, y start), (x end, y end))
            coords = ( (x*spaceWidth,0) , (x*spaceWidth,top))
            draw.line(coords,width=lineWidth,fill=lineColor)

        #Draws the border of the map.
        draw.rectangle([0,0,width-1,top-1],outline=lineColor,width=borderWidth)

        del draw

    def display(self):
        """Displays the map.
        """
        self.img.show()

if __name__=='__main__':

    import cProfile, pstats, io
    from pstats import SortKey

    pr = cProfile.Profile()
    pr.enable()
    
    map = Millow('sparse islands')
    map.toImage()
    map.addGrid((30,20))
    map.display()
    pr.disable()

    sortby = SortKey.CUMULATIVE
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())