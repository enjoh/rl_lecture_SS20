import numpy as np

"""
File containing general utility function used in WindyGridWorld.py and DynaQ.py
"""
# taken from https://stackoverflow.com/questions/42071597/numpy-argmax-random-tie-breaking
def rand_argmax(b, **kw):
    """ a random tie-breaking argmax"""
    bb = b == b.max()
    r = np.random.random(b.shape)
    return np.argmax(r * bb, **kw)
