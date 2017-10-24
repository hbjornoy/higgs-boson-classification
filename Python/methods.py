import numpy as np
from grad_loss import *
from dataprocessing import *

##Least squares using gradient descent
###########################################################################
def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    w = initial_w
    for n_iter in range(max_iters):
        gradient=compute_gradient(y,tx,w)
        w=w-gamma*gradient  
    loss=get_mse(y,tx,w)
    return loss, w


##Least squares using stochastic gradient descent
###########################################################################
def least_squares_SGD(y, tx, initial_w,max_iters, gamma):
    batch_size=1 #round(len(y)/100)
    w = initial_w
    for n_iter in range(max_iters):
        gradient=compute_stoch_gradient(y,tx,w,batch_size)
        w=w-gamma*gradient
    loss=get_mse(y,tx,w)
    return loss, w


## Least squares using normal equations
###########################################################################
def least_squares(y, tx): 
    trans_tx=np.transpose(tx)
    a=np.dot(trans_tx,tx)
    b=np.dot(trans_tx,y)
    w=np.linalg.solve(a,b)
    mse= get_mse(y,tx,w)
    return mse, w

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


## Logistic Regression using gradient descent
###########################################################################
def logistic_regression(y, tx, initial_w,max_iters, gamma):
        y=y.reshape(y.shape[0],1) #So that dimensions mathces
        w = initial_w
        for iter in range(max_iters):
            grad=calculate_gradient(y, tx, w)
            w=w-gamma*grad 
        loss=calculate_loss(y, tx, w)
        return loss, w


##Regularized Logistic Regression using gradient descent:
###########################################################################
def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    y=y.reshape(y.shape[0],1) #So that dimensions mathces
    w = initial_w
    for iter in range(max_iters):
        grad=calculate_gradient(y, tx, w)+ lambda_*w
        w=w-gamma*grad
    loss=calculate_loss(y, tx, w) + (lambda_/2)*np.linalg.norm(w,2)**2
    return loss, w


##Regularized Logistic Regression using NEWTON:
###########################################################################
def reg_logistic_regression_newton(y, tx, lambda_, initial_w, max_iters, gamma):
    y=y.reshape(y.shape[0],1) #So that dimensions mathces
    w = initial_w
    for iter in range(max_iters):
        hessian=calculate_hessian(y, tx, w)
        inv_hessian=np.linalg.inv(hessian)
        grad=calculate_gradient(y, tx, w)+ lambda_*w
        w=w-gamma*(np.dot(inv_hessian,grad))
    loss=calculate_loss(y, tx, w) + (lambda_/2)*np.linalg.norm(w,2)**2
    return loss, w
