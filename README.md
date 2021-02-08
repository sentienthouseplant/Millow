# Millow Map. (0.1.0)
A Python procedural map generator. 

#Features
The program accepts a specified map type (Continents, Dense Islands, etc) and generates a basic 1080x1920 rgb pillow Image object

#How-To
Currently there are 3 types of map:
-Continents.
-Dense Islands.
-Sparse Islands.

'''python
import Millow.Millow as Millow

map = Millow('map type') #Creates a Millow object with a given map type.
img = map.toImage() #Produces a pillow image object.
img.show() #Shows the generated map, one can use any pillow image methods on img. 
