# Millow (0.1.1)
A Python procedural map generator. 

# Features
The program accepts a specified map type (Continents, Dense Islands, etc) and generates a map.

# How-To
```python
import Millow.Millow as Millow

map = Millow('map type', mapSize=(4000,4000)) #Creates a Millow object with a given map type, and its 4000x4000 pixels.
map.generateBasic() #This creates a very basic green and blue map.
map.addHeight() #Adds height gradients to the map, indicates things like hills, etc.
map.addGrid((30,30)) #Adds a 30x30 grid over the top of the image. 
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
map.generateBasic()
map.addHeights()

map = Millow('sparse islands',mapSize=(1080,1920))
map.generateBasic()
map.addGrid(gridDensity=(10,10))
```
## Output.

### Continents.
![continents](https://github.com/Jackbytes/Millow-Map/blob/main/img/continents.png "Continents")

### Sparse Islands.
![Dense Islands](https://github.com/Jackbytes/Millow-Map/blob/main/img/sparseislands.png "Sparse Islands")

# To Do.

- [ ] Add water shading.
- [ ] Remove gray/black dot bug.
