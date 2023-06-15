import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
# from lr_utils import load_dataset
def load_dataset():
    train_dataset = h5py.File("C:/Users/da/Desktop/coding/class/introToCS/wk - 2/cat_dog/datasets/train_catvnoncat.h5", "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File("C:/Users/da/Desktop/coding/class/introToCS/wk - 2/cat_dog/datasets/test_catvnoncat.h5", "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

# %matplotlib inline


# Loading the data (cat/non-cat)
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

# Example of a picture
# index = 23
# plt.imshow(train_set_x_orig[index])
# plt.show()

# print("y = " + str(train_set_y[:, index]) + ", it's a '" + classes[np.squeeze(train_set_y[:, index])].decode("utf-8") +  "' picture.")

m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]

# print ("Number of training examples: m_train = " + str(m_train))
# print ("Number of testing examples: m_test = " + str(m_test))
# print ("Height/Width of each image: num_px = " + str(num_px))
# print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
# print ("train_set_x shape: " + str(train_set_x_orig.shape))
# print ("train_set_y shape: " + str(train_set_y.shape))
# print ("test_set_x shape: " + str(test_set_x_orig.shape))
# print ("test_set_y shape: " + str(test_set_y.shape))

train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

# print ("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
# print ("train_set_y shape: " + str(train_set_y.shape))
# print ("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
# print ("test_set_y shape: " + str(test_set_y.shape))
# print ("sanity check after reshaping: " + str(train_set_x_flatten[0:5,0]))


train_set_x = train_set_x_flatten / 255
test_set_x = test_set_x_flatten / 255

def sigmoid(z): 
    s = 1 / (1 + np.exp(-z))
    return s

def initialize_with_zeros(dim):
    w = np.zeros((dim, 1))
    b = 0
    assert(w.shape == (dim, 1)) # we assert that w.shape is (dim, 1) to make sure that the shape of w is correct
    assert(isinstance(b, float) or isinstance(b, int)) # we assert that b is a real number so that we can use float() function to convert b to a float number
    return w, b

def propagate(w, b, X, Y): 
    # Implement the cost function and its gradient for the propagation explained above

    # Arguments:
    # w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    # b -- bias, a scalar
    # X -- data of size (num_px * num_px * 3, number of examples)
    # Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

    # Return:
    # cost -- negative log-likelihood cost for logistic regression
    # dw -- gradient of the loss with respect to w, thus same shape as w
    # db -- gradient of the loss with respect to b, thus same shape as b
    
    # Tips:
    # - Write your code step by step for the propagation. np.log(), np.dot()

    m = X.shape[1]

    #Forward Propagation
    A = sigmoid(np.dot(w.T, X) + b)
    cost = np.sum(((-np.log(A) * Y)  - (np.log(1 - A) * (1 - Y)))) / m # compute cost
    
    #Backward Propagation
    dw = np.dot(X, (A - Y).T) / m
    db = np.sum(A - Y) / m

    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    grads = {"dw": dw,
                "db": db}
                
    return grads, cost


def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False): 
    costs = []
    # grads = {}
    for i in range(num_iterations):
        ## Cost and gradient calculation
        grads, cost = propagate(w, b, X, Y) # compute cost and gradient by using propagate() function that takes w, b, X, Y as input and calculates cost and gradient by using forward and backward propagation
        ## Retrieve derivatives from grads
        dw = grads["dw"] # retrieve dw from grads, this is the gradient of the loss with respect to w
        db = grads["db"] # retrieve db from grads, this is the gradient of the loss with respect to b
        ## update rule
        w = w - (learning_rate * dw) # we update w by using this formula: w = w - learning_rate * dw
        b = b - (learning_rate * db) # we update b by using this formula: b = b - learning_rate * db, this calculates the gradient descent for b by using the derivative of the loss with respect to b
       # Record the costs 
        if i % 100 == 0:
            costs.append(cost) # we record the cost every 100 iterations
        # Print the cost every 100 training iterations
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" %(i, cost))
    # update w and b by using the optimize() function. We want to minimize the cost function by using gradient descent, so we will use optimize() function to update w and b        
    params = {"w": w,
                "b": b}
    grads = {"dw": dw,
                "db": db}
    # we return params, grads, and costs so that we can use them to predict and plot the costs later
    return params, grads, costs

# params, grads, costs = optimize(w, b, X, Y, num_iterations= 100, learning_rate = 0.009, print_cost = False)
# print ("w = " + str(params["w"]))
# print ("b = " + str(params["b"]))
# print ("dw = " + str(grads["dw"]))
# print ("db = " + str(grads["db"]))
# print ("costs = " + str(costs))

# plt.plot(costs)
# plt.xlabel('Iterations (per hundreds)')
# plt.ylabel('Cost')
# plt.title('Cost reduction over time')
# plt.show()

def predict(w, b, X):
    # Computes the probability of a cat being present in the picture

    # Arguments:
    # w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    # b -- bias, a scalar
    # X -- data of size (num_px * num_px * 3, number of examples)

    # Returns:
    # Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X

    m = X.shape[1]
    Y_prediction = np.zeros((1, m)) # initialize Y_prediction as a numpy array (vector) of zeros of shape (1, m)
    w = w.reshape(X.shape[0], 1) # we reshape w and b by using w.reshape(X.shape[0], 1) so that we can compute Y_prediction by using np.dot(w.T, X) + b
    
    A = sigmoid(np.dot(w.T, X) + b) # compute vector "A" predicting the probabilities of a cat being present in the picture

    # Y_prediction = np.where(A > 0.5, 1, 0) # we use if/else statement to classify A[0, i] to be 1 (if A[0, i] > 0.5) or 0 (if A[0, i] <= 0.5)
    Y_prediction = (A >= 0.5) * 1 # we use if/else statement to classify A[0, i] to be 1 (if A[0, i] > 0.5) or 0 (if A[0, i] <= 0.5)


    assert(Y_prediction.shape == (1, m))

    return Y_prediction

# print("predictions = " + str(predict(w, b, X)))

def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
    # Builds the logistic regression model by calling the functions we have implemented previously
    #arguments:
    # X_train -- training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
    # Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
    # X_test -- test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
    # Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
    # num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
    # learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()

    # Returns:
    # d -- dictionary containing information about the model.

    # we initialize parameters with zeros by using initialize_with_zeros() function
    w, b = initialize_with_zeros(X_train.shape[0])

    # we calculate the cost and gradient for the current parameters by using propagate() function
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost = False)

    # we retrieve parameters w and b from dictionary "parameters"
    w = parameters["w"]
    b = parameters["b"]

    # we predict test/train set examples by using predict() function
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    # we print train/test errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    # we save dictionay "parameters" to be used for later use
    d = {"costs": costs,
            "Y_prediction_test": Y_prediction_test,
            "Y_prediction_train": Y_prediction_train,
            "w": w,
            "b": b,
            "learning_rate": learning_rate,
            "num_iterations": num_iterations}
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show() 
    
    
    return d

def visualize_weights(w):
    w_image = w.reshape(num_px, num_px, 3)  # reshape to original image dimensions

    # Adjust weights to be in the range 0-1
    w_min = np.min(w_image)
    w_max = np.max(w_image)
    w_image = (w_image - w_min) / (w_max - w_min)

    plt.imshow(w_image)
    plt.title('Visualization of the weights')
    plt.xlabel('Width')
    plt.ylabel('Height')
    plt.colorbar(label='Weight Value')
    plt.show()


d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = 0.005, print_cost = False)
visualize_weights(d["w"])
