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


##Sigmoid: Function used in logistic regression
###########################################################################
def sigmoid(t):
    a=np.exp(t)/(1+np.exp(t))
    return a
    


## Classify: Takes y from 1 and -1 to 1 and 0
###########################################################################
def classify(y):
    ty=np.copy(y)
    for i in range(len(y)):
        if y[i]==-1: 
            ty[i]=0
    return ty



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


## Log pred: 
## calculates the pobability and select y value 1 or -1
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



## Log pred acc: 
## given the correct values of y and the predicted values of y, it returns
## percentage and number of correct predictions
###########################################################################
def pred_acc(y,pred_y):
    counter=0
    for i in range(len(pred_y)):
        if pred_y[i]!= y[i]:
            counter=counter+1
    percent=counter/len(pred_y)
    return percent, counter



## Principal Component Analysis: 
## takes inn the data and desired number of dimensions, returns the transformed data,
## eigenvalues and eigenvectors of the covariance matrix. 
###########################################################################
def pca(data, dimensions = None):
    #standarizing the data:
    data -= np.mean(data, 0)
    data /= np.std(data, 0)
    
    #calculating the covariance, sigma:
    sigma = np.dot(data.T, data) / data.shape[0]
    
    #finding eigenvalues and vectors
    eig_val, eig_vec = np.linalg.eigh(sigma)
    
    #sorting them in order of deceasing eigenvalue. 
    #Choosing only the desired number of dimensions
    key = np.argsort(eig_val)[::-1][:dimensions]
    eig_val, eig_vec = eig_val[key], eig_vec[:, key]
    
    #tranforming the data using the principal components: 
    trans_data = np.dot(data, eig_vec)  
    return trans_data, eig_val, eig_vec


