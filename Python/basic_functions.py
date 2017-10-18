# Useful starting lines
import numpy as np
import matplotlib.pyplot as plt
from helpers import *

## Help function 1:computing the gradient
def compute_gradient(y, tx, w):
    N = len(y)
    error = (y - np.dot(tx,w))
    gradient=-(1/N)*np.dot(np.transpose(tx),error)
    return gradient

## Help funciton 2: calculating the mean squared error
def get_mse(y, tx, w):
    N = len(y)
    error = (y - np.dot(tx,w))
    mse = (1/(2*N)) * np.sum(np.square(error))
    return mse

## Help function 3: comcputing the stochastic gradient
def compute_stoch_gradient(y, tx, w, batch_size):
    for y_batch, tx_batch in batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
        gradient=compute_gradient(y_batch, tx_batch, w)
    return gradient 


## Help funciton 4: building polynomial matrix
def build_poly(x, degree):
    # this function should return the matrix formed
    # by applying the polynomial basis to the input data
    n,m=x.shape
    phi=np.zeros((n, m*(degree+1)))
    for k in range(m):
        for j in range(degree+1): 
            for i in range(n):
                phi[i,k*(m+1)+j]=x[i,k]**j
    return phi

## Help fucntion 5: 
def sigmoid(t):
    a=np.exp(t)/(1+np.exp(t))
    return a
    
## Help function 6: computing log loss: 
def calculate_loss(y, tx, w):
    """compute the cost by negative log likelihood."""
    sum_=0
    for i in range(len(y)):
        tr_x=np.transpose(tx[i,:])
        #print('this is the shape of w')
        #print(w.shape, tr_x.shape)
        #print(np.dot(tr_x,w).shape)
        sum_=sum_+np.log(1+np.exp(np.dot(tr_x,w)))-np.dot(y[i],np.dot(tr_x,w))
    return sum_
def classify(y):
    for i in range(len(y)):
        if y[i]==-1: 
            y[i]=0
    return y
## Help function 7: Standarize
def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x, axis=0)
    x = x - mean_x
    std_x = np.std(x, axis=0)
    x = x / std_x
    return x, mean_x, std_x
def normalize(x):
    n=x.shape[1]
    for i in range(n):
        max_=np.max(x[:,i])
        min_=np.min(x[:,i])
        x[:,i]=(x[:,i]-min_)/(max_-min_)
    return x


def calculate_gradient(y, tx, w):
    """compute the gradient of loss."""
    sigma=sigmoid(np.dot(tx,w))
    tr_tx=np.transpose(tx)
    grad= np.dot(tr_tx,(sigma-y))
    return grad

## Gradient Descent
def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        gradient=compute_gradient(y,tx,w)
        loss=get_mse(y,tx,w)
        w=w-gamma*gradient
        ws.append(w)
        losses.append(loss)

    return losses, ws


## Stochastic Gradient Descent
def least_squares_SGD(y, tx, initial_w,max_iters, gamma):
    batch_size=1 #round(len(y)/100)
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        gradient=compute_stoch_gradient(y,tx,w,batch_size)
        loss=get_mse(y,tx,w)
        w=w-gamma*gradient
        ws.append(w)
        losses.append(loss)
    return losses, ws, w


## Normal Equations
def least_squares(y, tx): 
    # returns mse, and optimal weights
    trans_tx=np.transpose(tx)
    a=np.dot(trans_tx,tx)
    b=np.dot(trans_tx,y)
    w=np.linalg.solve(a,b)
    mse= get_mse(y,tx,w)
    return mse, w
    

## Ridge Regression, Normal Equations
def ridge_regression(y, tx, lambda_):
    N,M=tx.shape
    I=np.identity(M)
    t_tx=np.transpose(tx)
    matrix_inv=np.linalg.inv(np.dot(t_tx,tx)+lambda_*2*N*I)
    w_ridge=np.dot(np.dot(matrix_inv,t_tx),y)
    rmse=np.sqrt(2*(get_mse(y,tx,w_ridge)+lambda_*np.linalg.norm(w_ridge,ord=2)**2))
    return w_ridge, rmse

def split_data(x, y, ratio, seed=1):
    """
    split the dataset based on the split ratio. If ratio is 0.8 
    you will have 80% of your data set dedicated to training 
    and the rest dedicated to testing
    """
    # set seed
    np.random.seed(seed)
    
    # split the data based on the given ratio: TODO
    n=len(x)
    proportion_in_training= round(n*ratio)
    indices = np.random.permutation(x.shape[0])
    training_idx, test_idx = indices[:proportion_in_training], indices[proportion_in_training:]
    training_x, test_x = x[training_idx], x[test_idx]
    training_y, test_y = y[training_idx], y[test_idx]
    return training_x, test_x, training_y, test_y

## Logistic Regression, GD or SGD
def logistic_regression(y, tx, initial_w,max_iters, gamma):
        y=y.reshape(y.shape[0],1) #Må ha dette for å få det til å fungere..
        w = initial_w
        losses=[]
        for iter in range(max_iters):
            loss=calculate_loss(y, tx, w)
            grad=calculate_gradient(y, tx, w)
            w=w-gamma*grad
            losses.append(loss) 
        return loss, w, losses


## Regularized logistic regression, GD or SGD
def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    y=y.reshape(y.shape[0],1) #Må ha dette for å få det til å fungere..
    w = initial_w
    losses=[]
    for iter in range(max_iters):
        loss=calculate_loss(y, tx, w) + lambda_/2*np.linalg.norm(w,2)**2
        grad=calculate_gradient(y, tx, w)+ lambda_*w
        w=w-gamma*grad
        losses.append(loss) 
    return loss, w, losses
