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

def softmax(x):
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis=1, keepdims=True)
    s = x_exp/x_sum
    return s

x = np.array([
    [9, 2, 5, 0, 0],
    [7, 5, 0, 0 ,0]])

print("softmax(x) = " + str(softmax(x)))

def L1(yhat, y):
    loss = np.sum(np.abs(yhat-y))
    return loss

yhat = np.array([.9, 0.2, 0.1, .4, .9])
y = np.array([1, 0, 0, 1, 1])

print("L1 = " + str(L1(yhat,y)))

def L2(yhat, y):
    loss = np.sum(np.power((yhat-y),2))
    return loss

print("L2 = " + str(L2(yhat,y)))

def vectorization():
    import time
    a = np.random.rand(1000000)
    b = np.random.rand(1000000)

    tic = time.time()
    c = np.dot(a,b)
    toc = time.time()

    print("Vectorized version: " + str(1000*(toc-tic)) + "ms")

    c = 0
    tic = time.time()
    for i in range(1000000):
        c += a[i] * b[i]
    toc = time.time()

    print("For loop: " + str(1000*(toc-tic)) + "ms")

vectorization()

def gradientDescent():
    def sigmoid(x):
        s = 1/(1+np.exp(-x))
        return s

    def initialize_with_zeros(dim):
        w = np.zeros((dim,1))
        b = 0
        return w, b

    def propagate(w, b, X, Y):
        m = X.shape[1]
        A = sigmoid(np.dot(w.T,X) + b)
        cost = (-1/m) * np.sum(Y*np.log(A) + (1-Y)*np.log(1-A))
        dw = (1/m) * np.dot(X, (A-Y).T)
        db = (1/m) * np.sum(A-Y)
        cost = np.squeeze(cost)
        grads = {"dw": dw,
                "db": db}
        return grads, cost

    def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
        costs = []
        for i in range(num_iterations):
            grads, cost = propagate(w, b, X, Y)
            dw = grads["dw"]
            db = grads["db"]
            w = w - learning_rate * dw
            b = b - learning_rate * db
            if i % 100 == 0:
                costs.append(cost)
            if print_cost and i % 100 == 0:
                print("Cost after iteration %i: %f" %(i, cost))
        params = {"w": w,
                "b": b}
        grads = {"dw": dw,
                "db": db}
        return params, grads, costs

    def predict(w, b, X):
        m = X.shape[1]
        Y_prediction = np.zeros((1,m))
        w = w.reshape(X.shape[0], 1)
        A = sigmoid(np.dot(w.T, X) + b)
        for i in range(A.shape[1]):
            if A[0,i] <= 0.5:
                Y_prediction[0,i] = 0
            else:
                Y_prediction[0,i] = 1
        return Y_prediction
    
    def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
        w, b = initialize_with_zeros(X_train.shape[0])
        parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
        w = parameters["w"]
        b = parameters["b"]
        Y_prediction_test = predict(w, b, X_test)
        Y_prediction_train = predict(w, b, X_train)
        print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
        print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))
        d = {"costs": costs,
            "Y_prediction_test": Y_prediction_test,
            "Y_prediction_train": Y_prediction_train,
            "w": w,
            "b": b,
            "learning_rate": learning_rate,
            "num_iterations": num_iterations}
        return d
    dim = 2
    w, b = initialize_with_zeros(2)
    print("w = " + str(w))
    print("b = " + str(b))
    # model().predict().optimize().propagate().initialize_with_zeros().sigmoid()
    

gradientDescent()

