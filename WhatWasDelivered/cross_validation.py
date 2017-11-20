import numpy as np
from implementations import*
from grad_loss import*
from dataprocessing import *
from proj1_helpers import *


def cross_validation(function_to_run, y, x, num_of_k_fold, *args):
    """
    Performes a "num_of_k_fold"-fold cross validation using the method "function_to_run"
    (a string with the name of a funciton), using the input data y and x, and additional 
    arguments (such as lambda, gamma, number of iteratioins) in *args.
    Returns 
    avg_loss: average loss 
    losses: a vecotr containing the loss for each fold
    avg_acc: avereage error in predictions 
    pred_acc_percents: error in predictions for each fold
    """
    
    losses = []
    pred_acc_percents = []
    
    k_indices = build_k_indices(y, num_of_k_fold, 1)
    
    for k in range(num_of_k_fold):
        
        x_test, y_test, x_tr, y_tr = split_k(x,y,k_indices, k)
        
        if(function_to_run.__name__ == "reg_logistic_regression"):

            lambda_ = args[0]
            initial_w = args[1]
            max_iters = args[2]
            gamma = args[3]
            
            weights, loss =  function_to_run(y_tr, x_tr, lambda_, initial_w, 
                                             max_iters, gamma) 
            pred_y = log_pred(x, weights) 
        
        elif(function_to_run.__name__ == "logistic_regression"):
            initial_w = args[0]
            max_iters = args[1]
            gamma = args[2]
            
            weights, loss =  function_to_run(y_tr, x_tr, initial_w, max_iters, gamma) 
            pred_y = log_pred(x, weights) 
            
        elif(function_to_run.__name__ == "ridge_regression"):
            
            lambda_ = args[0]
            
            weights, loss =ridge_regression(y, x, lambda_)  
            pred_y=predict_labels(weights, x)
            
        elif(function_to_run.__name__ == "least_squares_GD"):
                        
            weights = args[0]
            max_iters = args[1]
            gamma = args[2]  
             
            weights, loss = least_squares_GD(y, x, weights, max_iters, gamma)
            pred_y = predict_labels(weights, x)
            
        elif(function_to_run.__name__ == "least_squares_SGD"):
                        
            weights = args[0]
            max_iters = args[1]
            gamma = args[2]  
             
            weights, loss = least_squares_SGD(y, x, weights, max_iters, gamma)
            pred_y = predict_labels(weights, x)
            
        elif(function_to_run.__name__ == "least_squares"):
                         
            weights, loss = least_squares(y, x)
            pred_y = predict_labels(weights, x)
        
        losses.append(loss)

        pred_acc_percent, counter = pred_acc(y, pred_y)
        pred_acc_percents.append(pred_acc_percent)
     
    loss_sum = 0
    for loss in losses:
        loss_sum += loss
    avg_loss = loss_sum / len(losses)
    
    acc_sum = 0
    for acc in pred_acc_percents:
        acc_sum += acc
    avg_acc = acc_sum / len(pred_acc_percents)
    
    return avg_loss, losses, avg_acc, pred_acc_percents 


def build_k_indices(y, k_fold, seed):
    """
    given y, number of folds, "k_folds" and a seed, 
    the function returns the indicies for each of the "k-fold"
    folds
    """
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)


def split_k(x,y,k_indices, k):
    """
    Given x, y, "k_indices" as returned from the function build_k_indices
    and a spesific k, it returns test data containing only the k-th fold, and 
    training data containing all other folds. 
    """

    x_test, y_test= x[k_indices[k]], y[k_indices[k]]
    train_ind=np.delete(k_indices,k,0)
    train_ind=np.ravel(train_ind)
    x_tr, y_tr= x[train_ind], y[train_ind]

    return x_test, y_test, x_tr, y_tr

