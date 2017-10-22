import numpy as np
from helpers import * 
from dataprocessing import*
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
        #print('this is the shape of w')
        #print(w.shape, tr_x.shape)
        #print(np.dot(tr_x,w).shape)
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