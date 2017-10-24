import numpy as np
from methods import*
from grad_loss import*
from dataprocessing import *
from proj1_helpers import *


def cross_validation(function_to_run, y, x, num_of_k_fold, *args):
    
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
            
            loss, weights, losses =  function_to_run(y_tr, x_tr, lambda_, initial_w, 
                                                     max_iters, gamma) 
            pred_y = log_pred(x, weights) 
            
        elif(function_to_run.__name__ == "ridge_regression"):
            
            lambda_ = args[0]
            
            loss, weights =ridge_regression(y, x, lambda_)  
            pred_y=predict_labels(weights, x)
            
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
