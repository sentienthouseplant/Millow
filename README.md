# Millow (0.1.0)
A Python procedural map generator. 

# Features
The program accepts a specified map type (Continents, Dense Islands, etc) and generates a basic 1080x1920 pillow Image object

# How-To
```python
import Millow.Millow as Millow

map = Millow('map type') #Creates a Millow object with a given map type.
img = map.toImage() #Produces a pillow image object.
img.show() #Shows the generated map, one can use any pillow image methods on img. 
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
map = Millow('continents')
map.toImage().show()

map = Millow('dense islands')
map.toImage().show()

map = Millow('sparse islands')
map.toImage().show()
```
## Output.

### Continents.
![continents](https://github.com/Jackbytes/Millow-Map/blob/main/img/continents.png "Continents")

### Dense Islands.
![Dense Islands](https://github.com/Jackbytes/Millow-Map/blob/main/img/denseislands.png "Dense Islands")

### Sparse Islands.
![Sparse Islands](https://github.com/Jackbytes/Millow-Map/blob/main/img/sparseislands.png "Sparse Islands")

# To-Do.

- [ ] Add water shading.
- [ ] Add option for discrete gradients.
