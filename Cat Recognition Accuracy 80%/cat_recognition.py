import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
from dnn_app import *
from tkinter.filedialog import askopenfilename
from tkinter import ttk
from tkinter import *
from tkinter import filedialog

np.random.seed(1)
train_x_orig,train_y,test_x_orig,test_y,classes=load_data()
m_train = train_x_orig.shape[0]
numpx = train_x_orig.shape[1]
m_test = test_x_orig.shape[0]




train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T
train_x = train_x_flatten/255
test_x = test_x_flatten/255


n_x=12288
n_h=7
n_y=1
layer_dims=(n_x,n_h,n_y)

def two_layer_model(x,y,layer_dims,learning_rate=0.0075,num_iterations=3000,print_cost=False):
    np.random.seed(1)
    grads={}
    costs={}
    m=x.shape[1]
    (n_x,n_h,n_y)=layer_dims
    parameters=initialize_parameters(n_x,n_h,n_y)
    w1=parameters["W1"]
    b1=parameters["b1"]
    w2=parameters["W2"]
    b2=parameters["b2"]

    for i in range(0,num_iterations):
        a1,cache1=linear_activation_forward(x,w1,b1,activation='relu')
        a2,cache2=linear_activation_forward(a1,w2,b2,activation='sigmoid')
        cost=compute_cost(a2,y)
        da2 = - (np.divide(y, a2) - np.divide(1 - y, 1 - a2))
        da1 , dw2 , db2=linear_activation_backward(da2,cache2,activation='sigmoid')
        da0,dw1,db1=linear_activation_backward(da1,cache1,activation='relu')


        grads['dW1']=dw1
        grads['db1']=db1
        grads['dW2']=dw2
        grads['db2']=db2

        parameters=update_parameters(parameters,grads,learning_rate)

        w1=parameters['W1']
        b1=parameters['b1']
        w2=parameters['W2']
        b2=parameters['b2']

        if print_cost and i % 100 == 0:
            print("cost after iteration {} : {}  ".format(i,np.squeeze(cost)))
        if print_cost and i % 100==0:
            costs.setdefault(i,[]).append(cost)



    return parameters





layer_dims=[12288,20,7,5,1]
def L_layer_model(x,y,layers_dims,learning_rate=0.001,num_iterations=3000,print_cost=False):
    np.random.seed(1)
    costs=[]
    parameters=initialize_parameters_deep(layers_dims)

    for i in range(0,num_iterations):
        al,cache=L_model_forward(x,parameters)
        cost=compute_cost(al,y)
        grads=L_model_backward(al,y,cache)
        parameters=update_parameters(parameters,grads,learning_rate)

        if print_cost and i %100==0:
            print("cost after iterations %i %f" %(i,cost))
        if print_cost and i %100==0:
            costs.append(cost)

    return parameters

parameters = L_layer_model(train_x, train_y, layer_dims, num_iterations = 3500, print_cost = True)
my_label_y = [1]
filename=askopenfilename(filetypes=[("image files","*.jpg")])
my_image = filename
image = np.array(ndimage.imread(filename, flatten=False))
my_image = scipy.misc.imresize(image, size=(numpx,numpx)).reshape((numpx*numpx*3,1))
my_image = my_image/255.
my_predicted_image = predict(my_image, my_label_y, parameters)

plt.imshow(image)
print ("y = " + str(np.squeeze(my_predicted_image)) + ", your L-layer model predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")
