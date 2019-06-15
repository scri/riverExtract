
import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
from easy_ann_func import *
import cv2
img = np.array(Image.open("image/r.jpg"))
print(img.shape)
plt.imshow(img)

#选取训练数据  
img = cv2.imread("image/r.jpg")
cropped0 = img[2100:2200,0:100]
cropped1 = img[2100:2300,3600:3700]
cropped2 = img[2100:2300,6600:6700]
cropped3 = img[4200:4400,4500:4700]
cropped4 = img[2100:2200,100:300]
cropped5 = img[4200:4400,4700:4800]
plt.imshow(cropped5)
plt.imshow(cropped4)
plt.imshow(cropped3)
print(cropped3.shape)
#plt.imshow(cropped0)

plt.imshow(cropped0)
plt.imshow(cropped1)
print(cropped1.shape)
plt.imshow(cropped2)
print(cropped2.shape)
#手动选取训练数据d

def train_data():
    train_data_x = []
    train_data_y2 = []
    crop0 = np.array(cropped0).reshape(-1,3)
    crop1 = np.array(cropped1).reshape(-1,3)
    crop2 = np.array(cropped2).reshape(-1,3)
    crop3 = np.array(cropped3).reshape(-1,3)
    train_data_x0 = np.vstack((crop0,crop1))
    print(train_data_x0.shape)
    train_data_x1 = np.vstack((train_data_x0,crop2))
    print(train_data_x1.shape)
    train_data_x2 = np.vstack((train_data_x1,crop3))
    print(train_data_x2.shape)
    m = crop0.shape[0] + crop1.shape[0] + crop2.shape[0]
    print(m+crop3.shape[0])
    n = m + crop3.shape[0]
    train_data_y = np.zeros((1,n))
    for i in range(0,m):
        train_data_y[0,i] = 1
    
    return train_data_x2,train_data_y
def test_data():
    crop4 = np.array(cropped4).reshape(-1,3)
    crop5 = np.array(cropped5).reshape(-1,3)
    print(crop4.shape[0])
    test_data_x = np.vstack((crop4,crop5))
    m = crop4.shape[0]
    test_data_y = np.zeros((1,test_data_x.shape[0]))
    for j in range(0,m):
        test_data_y[0,j] = 1
    return test_data_x , test_data_y
y = [1,2,3,4]
y[0:4]
train_data_x, train_data_y = train_data()
test_data_x,test_data_y = test_data()
print(train_data_x.shape)
print(train_data_y.shape)
print(test_data_x.shape)
print(test_data_y.shape)
#print(train_data_y[0,370000-1])
print(test_data_y[0,20000-1])
train_x = train_data_x.T / 255.0
test_x = test_data_x.T / 255.0
print(test_x.shape)
print(train_x.shape)

layers_dims = [3,6,1]

def L_layer_model(X, Y, layers_dims, learning_rate = 0.01, num_iterations = 3000, print_cost=False):
    np.random.seed(1)
    costs = []                         # keep track of cost
    
    # Parameters initialization.
    ### START CODE HERE ###
    parameters = initialize_parameters_deep(layers_dims)
    ### END CODE HERE ###
    
    # Loop (gradient descent)
    for i in range(0, num_iterations):
        AL, caches = L_model_forward(X, parameters)
           cost = compute_cost(AL, Y)
   
        grads = L_model_backward(AL, Y, caches)
   
        parameters = update_parameters(parameters, grads, learning_rate)
     
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)
            
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters


# In[26]:


parameters = L_layer_model(train_x, train_data_y, layers_dims, learning_rate = 0.2, num_iterations = 4500, print_cost = True)

predictions_train = predict(train_x, train_data_y, parameters)

def predict(X, y, parameters):
    """
    This function is used to predict the results of a  L-layer neural network.
    
    Arguments:
    X -- data set of examples you would like to label
    parameters -- parameters of the trained model
    
    Returns:
    p -- predictions for the given dataset X
    """
    
    m = X.shape[1]
    n = len(parameters) // 2 # number of layers in the neural network
    p = np.zeros((1,m))
    
    # Forward propagation
    probas, caches = L_model_forward(X, parameters)

    
    # convert probas to 0/1 predictions
    for i in range(0, probas.shape[1]):
        if probas[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0
    
    #print results
    #print ("predictions: " + str(p))
    #print ("true labels: " + str(y))
    print("Accuracy: "  + str(np.sum((p == y)/m)))
        
    return p

predictions_train = predict(train_x, train_data_y, parameters)
predictions_test = predict(test_x, test_data_y, parameters)

img_flatten = img.reshape(-1,3).T / 255.0
print(img_flatten.shape)
print(predictions_test.shape)
def predict_img(X,parameters):
  
    
    m = X.shape[1]
    n = len(parameters) // 2 # number of layers in the neural network
    p = np.zeros((1,m))
    
    probas, caches = L_model_forward(X, parameters)

    
    # convert probas to 0/1 predictions
    for i in range(0, probas.shape[1]):
        if probas[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0
        
    return p
img_prediction = predict_img(img_flatten ,parameters)
print(img_prediction.reshape(img.shape[0],img.shape[1]))
img_river = img_prediction.reshape(img.shape[0],img.shape[1]) *50
plt.imshow(img_river)
import cv2
kernel = np.ones((11,11),np.uint8)  
erosion1 = cv2.erode(img_river,kernel,iterations = 1)
erosion2 = cv2.erode(erosion1,kernel,iterations = 1)
plt.imshow(erosion2)

dilation = cv2.dilate(erosion2,kernel,iterations = 1)
plt.imshow(dilation)

