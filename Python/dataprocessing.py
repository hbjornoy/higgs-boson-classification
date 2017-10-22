
from helpers import * 
import numpy as np

## Buld poly: builds a polynomial of degree "degree"
###########################################################################
def build_poly(x, degree):
    # this function should return the matrix formed
    # by applying the polynomial basis to the input data
    n,m=x.shape
    phi=np.zeros((n, m*(degree+1)))
    for k in range(m):
        for j in range(degree+1): 
            for i in range(n):
                #phi[i,k*(m+1)+j]=x[i,k]**j
                phi[i,k*(degree+1)+j]=x[i,k]**j
    return phi
###########################################################################

##Sigmoid: Function used in logistic regression
###########################################################################
def sigmoid(t):
    a=np.exp(t)/(1+np.exp(t))
    return a
    
###########################################################################

## Classify: Takes y from 1 and -1 to 1 and 0
###########################################################################
def classify(y):
    ty=np.copy(y)
    for i in range(len(y)):
        if y[i]==-1: 
            ty[i]=0
    return ty
###########################################################################

##Standardize: Standardizes the data
###########################################################################
def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x, axis=0)
    x = x - mean_x
    std_x = np.std(x, axis=0)
    x = x / std_x
    return x, mean_x, std_x
###########################################################################


##Nomralize: normalizes the data
###########################################################################
def normalize(x):
    n=x.shape[1]
    for i in range(n):
        max_=np.max(x[:,i])
        min_=np.min(x[:,i])
        if max_ != min_:
            x[:,i]=(x[:,i]-min_)/(max_-min_)
    return x
###########################################################################


##Split data: taken from ML course. 
###########################################################################
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
###########################################################################


##Clean data: Takes care of the not reccorded values in a number of different ways
###########################################################################
def clean_data(data_x, replace_no_measure_with_mean = False, replace_no_measure_with_median = False):
    
    data=np.copy(data_x)
    
    if(replace_no_measure_with_mean):
        
        rows, columns = data.shape
        column_means = np.zeros(30)

        #Manually counting mean for each column without -999 values. 
        for col in range(columns):
            c_count = 0
            c_sum = 0
            for row in range(rows - 1):
                data_point = data[row][col]
                
                if(data_point != -999):
                    c_count += 1
                    c_sum += data_point
            
            mean_mean = c_sum / c_count            
            column_means[col] = mean_mean
        
        #Replacing -999 values with column mean
        for col in range(columns - 1):
            for row in range(rows - 1):
                
                data_point = data[row][col]
                
                if(data_point == -999):
                    data[row][col] = column_means[col]  
                              
    
    elif(replace_no_measure_with_median):
                
        row, columns = data.shape
            
        column_medians = []
            
        for col in range(columns): 
            
            valid_points = []
            
            for row in range(row - 1): 
                
                data_point = data[row][col]
                
                if(data_point != -999):
                    
                    valid_points.append(data_point)
        
            median = np.median(valid_points)
            column_medians.append(median)
            
        for col in range(columns - 1):
            for row in range(row - 1):
                
                data_point = data[row][col]
                
                if(data_point == -999):
                    data[row][col] = column_medians[col]  
    
    else:
        data[data == -999] = 0
    
    return data
###########################################################################

##Log pred: calculates the pobability and select y value 1 or -1
###########################################################################
def log_pred(tx,w):
    probability=sigmoid(np.dot(tx,w))
    pred_y=np.zeros((len(probability)))
    for i in range(len(probability)):
        if probability[i]>0.5:
            pred_y[i]=1
        else:
            pred_y[i]=0

    return pred_y
###########################################################################

##Log pred acc: given the correct values of y and the predicted values of y, it returns
##  percentage and number of correct predictions
###########################################################################
def log_pred_acc(y,pred_y):
    counter=0
    for i in range(len(pred_y)):
        
        if pred_y[i]!= y[i]:
            counter=counter+1
    percent=counter/len(pred_y)
    return percent, counter
###########################################################################

#MÅ ENDRES LITT PÅ SÅ DEN IKKE BLIR TATT I PLAGIAT.. 
###########################################################################
def pca(data, pc_count = None):
    """
    Principal component analysis using eigenvalues
    note: this mean-centers and auto-scales the data (in-place)
    """
    data -= np.mean(data, 0)
    data /= np.std(data, 0)
    """
    Covariance matrix
    note: specifically for mean-centered data
    note: numpy's `cov` uses N-1 as normalization
    """
    C = np.dot(data.T, data) / data.shape[0]
    E, V = np.linalg.eigh(C)
    key = np.argsort(E)[::-1][:pc_count]
    E, V = E[key], V[:, key]
    U = np.dot(data, V)  # used to be dot(V.T, data.T).T
    return U, E, V
###########################################################################

##Taking inverse log of non negative data: 
###########################################################################
def inverse_log_non_neg(x):
    tx=np.copy(x)
    inv_log_cols = (0,1,2,3,4,5,7,8,9,10,12,13,16,19,21,23,26)
    
    tx[:,inv_log_cols]=np.log(1 / (1 + tx[:, inv_log_cols]))
    return tx
###########################################################################