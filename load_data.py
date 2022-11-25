from quickdraw import QuickDrawDataGroup
import numpy as np


max_d = 10000

airplanes = QuickDrawDataGroup('airplane', max_drawings=max_d)
castle = QuickDrawDataGroup('castle', max_drawings=max_d)
dragon = QuickDrawDataGroup('dragon', max_drawings=max_d)
duck = QuickDrawDataGroup('duck', max_drawings=max_d)
fork = QuickDrawDataGroup('fork', max_drawings=max_d)
hexagon = QuickDrawDataGroup('hexagon', max_drawings=max_d)
key = QuickDrawDataGroup('key', max_drawings=max_d)
mountain = QuickDrawDataGroup('mountain', max_drawings=max_d)
octopus = QuickDrawDataGroup('octopus', max_drawings=max_d)
pizza = QuickDrawDataGroup('pizza', max_drawings=max_d)
star = QuickDrawDataGroup('star', max_drawings=max_d)
submarine = QuickDrawDataGroup('sword', max_drawings=max_d)
sun = QuickDrawDataGroup('sun', max_drawings=max_d)
tree = QuickDrawDataGroup('tree', max_drawings=max_d)
dog = QuickDrawDataGroup('dog', max_drawings=max_d)


img_data = np.array([[np.array(drawing.get_image().resize((224, 224)).convert('L')) for drawing in c.drawings] for c in [airplanes, castle, dragon,
                                                                                          duck, fork, hexagon,
                                                                                          key, mountain, octopus,
                                                                                          pizza, star, submarine,
                                                                                          sun, tree, dog]]
                    )

img_classes = np.array([[drawing.name for drawing in c.drawings] for c in [airplanes, castle, dragon, duck, fork,
                                                                             hexagon, key, mountain, octopus, pizza,
                                                                             star, submarine, sun, tree, dog]]
                       )


print(img_data.shape)
print(img_data[0][0].shape)

np.save('img_data.npy', img_data)
np.save('img_classes.npy', img_classes)





