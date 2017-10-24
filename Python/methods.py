import numpy as np
from grad_loss import *
from dataprocessing import *


##Least squares using gradient descent
###########################################################################
def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    w = initial_w
    for n_iter in range(max_iters):
        gradient=compute_gradient(y,tx,w)
        loss=get_mse(y,tx,w)
        w=w-gamma*gradient        
    return loss, w
###########################################################################


##Least squares using stochastic gradient descent
###########################################################################
def least_squares_SGD(y, tx, initial_w,max_iters, gamma):
    batch_size=1 #round(len(y)/100)
    w = initial_w
    for n_iter in range(max_iters):
        gradient=compute_stoch_gradient(y,tx,w,batch_size)
        loss=get_mse(y,tx,w)
        w=w-gamma*gradient
    return loss, w
###########################################################################


## Least squares using normal equations
###########################################################################
def least_squares(y, tx): 
    # returns mse, and optimal weights
    trans_tx=np.transpose(tx)
    a=np.dot(trans_tx,tx)
    b=np.dot(trans_tx,y)
    w=np.linalg.solve(a,b)
    mse= get_mse(y,tx,w)
    
    return mse, w
###########################################################################

## Ridge regression:
###########################################################################
def ridge_regression(y, tx, lambda_):
    N,M=tx.shape
    I=np.identity(M)
    t_tx=np.transpose(tx)
    matrix_inv=np.linalg.inv(np.dot(t_tx,tx)+lambda_*2*N*I)
    w=np.dot(np.dot(matrix_inv,t_tx),y)
    rmse=np.sqrt(2*(get_mse(y,tx,w)+lambda_*np.linalg.norm(w,ord=2)**2))
    return rmse,w
###########################################################################

## Alternative Ridge regression:
###########################################################################
def alt_ridge_regression(y, tx, lambda_):
    N,M=tx.shape
    I=np.identity(M)
    t_tx=np.transpose(tx)
    a=(np.dot(t_tx,tx)+lambda_*2*N*I)
    b=np.dot(t_tx,y)
    w=np.linalg.solve(a,b)
    rmse=np.sqrt(2*(get_mse(y,tx,w)+lambda_*np.linalg.norm(w,ord=2)**2))
    return rmse,w
###########################################################################


## Logistic Regression using gradient descent
###########################################################################
def logistic_regression(y, tx, initial_w,max_iters, gamma):
        y=y.reshape(y.shape[0],1) #So that dimensions matches
        w = initial_w
        for iter in range(max_iters):
            loss=calculate_loss(y, tx, w)
            grad=calculate_gradient(y, tx, w)
            w=w-gamma*grad 
        return loss, w
###########################################################################


##Regularized Logistic Regression using gradient descent:
###########################################################################
def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    y=y.reshape(y.shape[0],1) #So that dimensions mathces
    w = initial_w
    for iter in range(max_iters):
        loss=calculate_loss(y, tx, w) + (lambda_/2)*np.linalg.norm(w,2)**2
        grad=calculate_gradient(y, tx, w)+ lambda_*w
        w=w-gamma*grad
    return loss, w
###########################################################################