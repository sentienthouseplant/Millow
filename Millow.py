# Outcomes of project.
#
# - Produce a png of a map with various properties stipulated by the user.


from PIL import Image, ImageDraw
import numpy as np
from scipy.ndimage import zoom, gaussian_filter

# Constants used--------------------------------------------
GLOBAL_CRIT_PROB = 0.59274621


# The functions---------------------------------------------

# This function produces a example of percolation. Please see the wikipedia article for more infomation.
# At a special occupation probability we get holes of all sizes (fractals). This would be a good candidate
# for an initial noise array to used for the map.
def noise_array(sizeTuple):
    prob = GLOBAL_CRIT_PROB
    ar = np.random.choice([0, 1], p=[1 - prob, prob], size=sizeTuple)
    return ar


# This function will smooth the resulting noise from the noise_array. This essentially yields a map with some
# basic landmasses. This is based on the cave generating cellular automata seen in many games.
def cellular_smooth(
    ar: np.array, overPop: int, underPop: int, bornPop: int, steps: int, borderValue=1
) -> np.array:

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
def height_array(size: (int, int), volat: float, noiseProb: float):
    """Returns a height profile given a 2-tuple and volatility."""

    if len(size) != 2:
        return "size must be 2D."

    rng = np.random.default_rng()  # A local rng used in this function.

    height_array = np.zeros(size)  # Initialises the array with zeros.

    height_array[0, 0] = rng.uniform(
        low=-1, high=1
    )  # The first value is given some initial value from U[0,1].

    height_array[0, 1] = rng.uniform(low=-1, high=1)

    height_array[1, 0] = rng.uniform(low=-1, high=1)

    for x0 in range(2, size[1]):  # Iterates over the first row, initialising it.
        if (
            rng.uniform() < noiseProb
        ):  # This is the chance that some independent randomness is introduced.
            height_array[0, x0] = rng.uniform(low=-1, high=1)
        else:
            uni = rng.uniform(low=-1, high=1)
            point1 = height_array[
                0, x0 - 2
            ]  # The value 2 places back in the first row.
            point2 = height_array[0, x0 - 1]  # The value 1 place back in the first row.
            height_array[0, x0] = (
                (point1 + point2) / 2
            ) + volat * uni  # Calulates the midpoint and adds a weighted U[-1,1].
        # This should add some dependence that results in sequences of similar values.

    for y0 in range(
        2, size[0]
    ):  # See above for loop. This repeats the above but for the first column.
        if rng.uniform() < noiseProb:
            height_array[y0, 0] = rng.uniform(low=-1, high=1)
        else:
            uni = rng.uniform(low=-1, high=1)
            point1 = height_array[y0 - 2, 0]
            point2 = height_array[y0 - 1, 0]
            height_array[y0, 0] = ((point1 + point2) / 2) + volat * uni

    for y in range(1, size[0]):  #
        for x in range(1, size[1]):
            if rng.uniform() < noiseProb:
                height_array[y, x] = rng.uniform(low=-1, high=1)
            else:
                uni = rng.uniform(low=-1, high=1)
                point1 = height_array[y - 1, x]
                point2 = height_array[y, x - 1]
                height_array[y, x] = ((point1 + point2) / 2) + volat * uni

    return height_array


def color_strat(alpha_array: np.array, ranges: list, colors: list) -> np.array:
    """Produces a rgba array with colors stratified according to the ranges given."""

    if not (len(colors) == len(ranges)):
        raise Exception("Each color in colors list must have a corresponding range.")

    red_array = green_array = blue_array = alpha_array

    rValues = [i[0] for i in colors]

    gValues = [i[1] for i in colors]

    bValues = [i[2] for i in colors]

    (a, b) = alpha_array.shape  # Unpacks size of a, a two dimensional array.

    resultArray = np.full(
        (a, b, 4), [255, 255, 255, 0], dtype=np.uint8
    )  # Creates a rgba array.

    counter = 0

    for low in ranges:
        red_array[(low <= a)] = rValues[counter]
        green_array[(low <= a)] = gValues[counter]
        blue_array[(low <= a)] = bValues[counter]

        counter = counter + 1

    resultArray[:, :, 0] = red_array
    resultArray[:, :, 1] = green_array
    resultArray[:, :, 2] = blue_array
    resultArray[:, :, 3] = alpha_array

    img = Image.fromarray(resultArray)
    img.show()

    return resultArray


class Millow:
    def __init__(self, given_map_type: str, map_size: (int, int) = (1080, 1920)):
        """Initiates the millow object.

        Parameters
        ----------
        given_map_type : A string from the list of possible map types.

            Specifies the type of map to generate.

        map_size: A tuple (Height,Width) which specifies the map size.

        Raises
        ------
        Exception
            Generates an exception if the given map type is not in the list of map types.
        """

        possible_map_types = [
            "continents",
            "dense islands",
            "sparse islands",
        ]  # Possible map types, will be updated as I find more.
        if type(given_map_type) != str:
            raise TypeError(
                'The map type must be a string and one of the possible options.'
            )

        if given_map_type not in possible_map_types:
            raise ValueError(
                "The map type must be one of the possible options."
            )  # Raises an exception if the user
        # attempts to give a invalid choice of map type.

        # The pixel size of the final image, currently not variable. Will be in future versions.
        self.size = map_size

        self.map_type = given_map_type  # Sets the map type. Currently unused.

        self.map_type_dict = {
            "continents": (9, 4, 6, 10, 1),
            "dense islands": (7, 1, 9, 15, 0),
            "sparse islands": (9, 2, 6, 18, 1),
        }
        # A dictionary which maps the users selected map type to the inputs of cellular_smooth. Inputs discovered by
        # experimentation.

        self.rawsGenerated = False  # Used to see if a map has been generated.

    def generate_basic(self):

        """Generates a basic array with land and water."""

        raw_map = noise_array(
            (int(self.size[0] / 50), int(self.size[1] / 50))
        )  # Size is divided by 50 since it will be zoomed by this
        # amount to create smooth features. This uses percolation at the critical probability for features of all sizes.

        (a, b, c, d, e) = self.map_type_dict[
            self.map_type
        ]  # Unpacks the constants used for the smoothing procedure, which
        # produces the desired map type.

        raw_map = cellular_smooth(
            raw_map, a, b, c, d, borderValue=e
        )  # See above for an explanation of a,b,c,d,e. This smooths the noise
        # produced by noise_array.

        raw_map = zoom(
            raw_map, 50
        )  # Zooming the array restores it to the intended pixel dimensions and smooths out
        # the roughness.

        raw_map = raw_map[
            0 : self.size[0], 0 : self.size[1]
        ]  # Trims any extra entries possibly caused by rounding.

        self.raw_map = raw_map  # Saves the generated smooth map array.

        colour_map = np.zeros(
            [self.size[0], self.size[1], 4], dtype=np.uint8
        )  # Initialises the RGBA array.

        colour_map[self.raw_map == 0] = [79, 76, 176, 255]  # Colours the 'water'.

        colour_map[self.raw_map == 1] = [159, 193, 100, 255]  # Colours the 'earth'.

        # Generates the basic color map.
        self.img = Image.fromarray(colour_map)

    def add_height(self):

        """Adds height levels to the raw_map property."""

        alpha_array = noise_array(
            (int(self.size[0] / 5), int(self.size[1] / 5))
        )  # Size is divided by 50 since it will be zoomed by this
        # amount to create smooth features. This uses percolation at the critical probability for features of all sizes.

        alpha_array = cellular_smooth(
            alpha_array, 9, 2, 6, 17, borderValue=0
        )  # Dense islands.

        alpha_array = zoom(
            alpha_array, 5
        )  # Zooming the array restores it to the intended pixel dimensions and smooths out
        # the roughness.

        alpha_array = alpha_array[
            0 : self.size[0], 0 : self.size[1]
        ]  # Trims any extra entries possibly caused by rounding.

        alpha_array = alpha_array.astype(
            np.float32
        )  # Changes the datatype so that Gaussian blur and divison work with the
        # desired accuracy.

        alpha_array[
            self.raw_map == 0
        ] = 0  # This masks the height array, so only value which correspond to '1's in the
        # raw_map are kept. The rest are set to zero. Thus zero is 'sea level'.

        alpha_array = gaussian_filter(alpha_array, 20)  # Blurs the image significantly.

        alpha_array[
            self.raw_map == 0
        ] = 0  # Remasks the array, now there is some coastal smoothing.

        alpha_array = (
            (alpha_array - np.amin(alpha_array)) / np.amax(alpha_array)
        ) * 255  # All values will be in the range (0,255).

        alpha_array = alpha_array.astype(np.uint8)

        # Gradient Heights

        red_array = np.interp(alpha_array, [0, 200, 255], [159, 252, 253])
        #
        green_array = np.interp(alpha_array, [0, 200, 255], [193, 234, 250])
        #
        blue_array = np.interp(alpha_array, [0, 200, 255], [100, 116, 212])

        raw_height = np.full(
            (self.size[0], self.size[1], 4), [255, 255, 255, 0], dtype=np.uint8
        )  # Produces a black RGBA array with zero opacity.

        raw_height[
            :, :, 3
        ] = alpha_array  # Sets the opaticy of the raw_height array to the one we generated.
        raw_height[:, :, 0] = red_array
        raw_height[:, :, 1] = green_array
        raw_height[:, :, 2] = blue_array

        self.raw_height = raw_height

        # Generates the heights image
        height_image = Image.fromarray(self.raw_height)

        # Composites the heights over the raw images.
        self.img = Image.alpha_composite(self.img, height_image)

    def add_grid(
        self,
        grid_size: (int, int),
        line_width: int = 2,
        line_color="white",
        border_width: int = 3,
    ):
        """Adds a grid to the map image object.

        Parameters
        ----------
        grid_size : A 2-tuple of ints.
            (Number of squares horizonially, number of squares vertically)
        line_width : Int, optional
            The width of the grid lines in pixels, by default 2.
        line_color : A pillow color, optional
            The color of the lines used for the grid, by default 'white'.
        border_width: Int, optional
            The width of the border of the whole map, by default 3.
        """

        draw = ImageDraw.Draw(self.img)

        # Gets the size of the generated image.
        width, top = self.img.size

        # Generates the spacing of the lines.
        space_width = width / grid_size[0]
        space_height = top / grid_size[1]

        # Draws the horizonial lines
        for y in range(0, grid_size[1]):

            # Line Coords ((x start, y start), (x end, y end))
            coords = ((0, y * space_height), (width, y * space_height))
            draw.line(coords, width=line_width, fill=line_color)

        # Draws the vertical lines
        for x in range(0, grid_size[0]):

            # Line Coords ((x start, y start), (x end, y end))
            coords = ((x * space_width, 0), (x * space_width, top))
            draw.line(coords, width=line_width, fill=line_color)

        # Draws the border of the map.
        draw.rectangle([0, 0, width - 1, top - 1], outline=line_color, width=border_width)

        del draw

    def display(self):
        """Displays the map."""
        self.img.show()


if __name__ == "__main__":

    map = Millow("sparse islands", map_size=(1080, 1920))
    map.generate_basic()
    map.add_height()
    map.display()