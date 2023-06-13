import numpy as np # this means you can access numpy functions by writing np.function() instead of numpy.function()
import math
# def basic_sigmoid(x):
#     s = 1/(1+math.exp(-x))
#     return s

def sigmoid(x):
    # s = 1/1(np.exp(-x) + 1)
    s = 1/(np.exp(-x) + 1)
    return s

x = np.array([19982839928,2,3])
print(sigmoid(x))
def sigmoid_derivative(x):
    s = sigmoid(x)
    ds = s * (1 - s)
    return ds  


def img2vec(img):
    v = img.reshape(img.shape[0] * img.shape[1] * img.shape[2], 1)
    return v 

x = np.array([
    [[0.67826139, 0.29380381],
    [0.90714982, 0.52835647],
    [0.4215251 , 0.45017551]],

    [[0.92814219, 0.96677647],
    [0.85304703, 0.52351845],
    [0.19981397, 0.27417313]],

    [[0.60659855, 0.00533165],
    [0.10820313, 0.49978937],
    [0.34144279, 0.94630077]]
    ])

# print(img2vec(x))

def normalizeRows(x):
    #compute x_norm as the norm 2 of x. Use np.linalg.norm(..., ord = 2, axis = ..., keepdims = True)
    x_norm = np.linalg.norm(x, axis=1, keepdims=True)
    
    #divide x by its norm.
    x = x/x_norm
    return x

x = np.array([
    [0, 3, 4],
    [1, 6, 4]])

print("normalizeRows(x) = " + str(normalizeRows(x)))
