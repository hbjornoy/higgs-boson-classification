import numpy as np
from dataprocessing import*
from proj1_helpers import *

###########################################################################
##Computing loss
###########################################################################

##Computing loss MSE:
###########################################################################
def get_mse(y, tx, w):
    N = len(y)
    error = (y - np.dot(tx,w))
    mse = (1/(2*N)) * np.sum(np.square(error))
    return mse
###########################################################################
##Computing loss logistic:
###########################################################################
def calculate_loss(y, tx, w):
    """compute the cost by negative log likelihood."""
    sum_=0
    for i in range(len(y)):
        tr_x=np.transpose(tx[i,:])
        sum_=sum_+np.log(1+np.exp(np.dot(tr_x,w)))-np.dot(y[i],np.dot(tr_x,w))
    return sum_
###########################################################################



###########################################################################
##Computing Gradients
###########################################################################

##Computing gradient MSE: 
###########################################################################
def compute_gradient(y, tx, w):
    N = len(y)
    y=y.reshape(y.shape[0],1)
    error = (y - np.dot(tx,w))
    gradient=-(1/N)*np.dot(np.transpose(tx),error)
    return gradient
###########################################################################
##Computing gradient stochastic MSE: 
###########################################################################
def compute_stoch_gradient(y, tx, w, batch_size):
    for y_batch, tx_batch in batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
        gradient=compute_gradient(y_batch, tx_batch, w)
    return gradient 
###########################################################################
##Computing gradient logistic: 
###########################################################################
def calculate_gradient(y, tx, w):
    """compute the gradient of loss."""
    sigma=sigmoid(np.dot(tx,w))
    tr_tx=np.transpose(tx)
    grad= np.dot(tr_tx,(sigma-y))
    return grad
###########################################################################


####HAR BARE PRØVD NOE, MEN SER IKKE UT SOM OM DET FUNGERER... 
def calculate_hessian(y, tx, w):
    """return the hessian of the loss function."""
    # *************************************************
    # calculate hessian: TODO
    xw = np.dot(tx,w)
    xT = np.transpose(tx)
    s = sigmoid(xw)*(1-sigmoid(xw))
    S = np.diagflat(s)
    H = np.dot(xT, np.dot(S,tx))
    return  H
    # *************************************************