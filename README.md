# Millow (0.1.1)
A Python procedural map generator. 

# Features
The program accepts a specified map type (Continents, Dense Islands, etc) and generates a map.

# How-To
```python
import Millow.Millow as Millow

map = Millow('map type', mapSize=(4000,4000)) #Creates a Millow object with a given map type, and its 4000x4000 pixels.
map.generate_basic() #This creates a very basic green and blue map.
map.add_height() #Adds height gradients to the map, indicates things like hills, etc.
map.add_grid((30,30)) #Adds a 30x30 grid over the top of the image. 
map.display() #Displays the map.
myPillowImage = map.img #The Pil(low) image object.
```


Currently there are the following map types:
- 'continents'
- 'dense islands'
- 'sparse islands'

# Requirements 

- scipy
- pillow
- numpy

# Examples.

## Input.
```python
map = Millow('continents',mapSize=(1000,1000))
map.generate_basic()
map.add_heights()

map = Millow('sparse islands',mapSize=(1080,1920))
map.generate_basic()
map.add_grid(grid_size=(10,10))
```
## Output.

### Continents.
![continents](https://github.com/Jackbytes/Millow-Map/blob/main/img/continents.png "Continents")

### Sparse Islands.
![Dense Islands](https://github.com/Jackbytes/Millow-Map/blob/main/img/sparseislands.png "Sparse Islands")
