# Useful starting lines
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
%load_ext autoreload
%autoreload 2
from helpers import *

## Help function 1:computing the gradient
def compute_gradient(y, tx, w):
    N = len(y)
    error = (y - np.dot(tx,w))
    gradient=-(1/N)*np.dot(np.transpose(tx),error)
    return gradient

## Help funciton 2: calculating the mean square error
def get_mse(y, tx, w):
    N = len(y)
    error = (y - np.dot(tx,w))
    mse = (1/(2*N)) * np.sum(np.square(error))
    return mse

## Help function 3: comcputing the stochastic gradient
## Veldig fint om noen har en mer elegant løsning
def compute_stoch_gradient(y, tx, w, batch_size):
    
    
    #FORSØK PÅ "swish" 
    
    for y_batch, tx_batch in batch_iter(y, tx, batch_size, max_iters, shuffle=True):
        
        gradient=compute_gradient(y_batch, tx_batch, w)

    
    #ORIGINAL FRA HEDDA SOM VIRKER HELT GARRA 

    #batches = batch_iter(y, tx, batch_size, num_batches=1, shuffle=True)
    #batch=next(batches) ## fant ingen bedre måter å gjøre dette på.. 

    #y_batch=batch[0]
    #tx_batch=batch[1]
    #gradient=compute_gradient(y_batch, tx_batch, w)
    
    return gradient 


## Help funciton 4: building polynomial matrix
def build_poly(x, degree):
    # this function should return the matrix formed
    # by applying the polynomial basis to the input data
    n=len(x)
    phi=np.zeros((n, degree+1))
    for j in range(degree+1): 
        for i in range(n):
            phi[i,j]=x[i]**j
    return phi


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
        #print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
        #    bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses, ws


## Stochastic Gradient Descent
def least_squares_SGD(y, tx, initial_w,max_iters, gamma):
    batch_size=1
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        gradient=compute_stoch_gradient(y,tx,w)
        loss=get_mse(y,tx,w)
        w=w-gamma*gradient
        ws.append(w)
        losses.append(loss)
        #print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
        #    bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses, ws


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
    rmse=np.sqrt(2*(compute_loss(y,tx,w_ridge)+lambda_*np.linalg.norm(w_ridge,ord=2)**2))
    return w_ridge, rmse


## Logistic Regression, GD or SGD
def logistic regression(y, tx, initial_w,max_iters, gamma):
    

## Regularized logistic regression, GD or SGD
def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma)
