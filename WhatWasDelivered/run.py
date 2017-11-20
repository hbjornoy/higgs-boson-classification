# standard libraries
import numpy as np
import matplotlib.pyplot as plt
import time


# own functions
import proj1_helpers as P1H
import dataprocessing as DP
import implementations as ME
import cross_validation as CV
from grad_loss import*

#constants
train_path = 'train.csv'
test_path = 'test.csv'

#importing data
orig_y, orig_x, orig_ids = load_csv_data(train_path, sub_sample=False) 
pred_y, pred_x, pred_ids = load_csv_data(test_path, sub_sample=False)

#Stacking training and test data together to easily preform the same transformations on both sets
all_x = np.vstack((orig_x, pred_x))

# value we split the all_x on before testing
split_coord = len(orig_y)

# To provide clarity to which data that is processed
x = np.copy(all_x)

#cleaning the data: replacing -999 with the median
clean_medi = DP.clean_data(x, replace_no_measure_with_median=True)

#Normalizing:
normalized=DP.normalize(clean_medi)

#Performing PCA, keeping 29 dimensions
pca=DP.pca(normalized,29)[0]

#setting hyperparameters:
degree=7
lambdas=np.logspace(-9,1,15)
lambda_= lambdas[4]

#creating polynomial:
phi=DP.build_poly(pca,degree)
#Determining number of folds in cross validation:
k_folds=5

#performing cross validation:
avg_loss, losses, avg_preds, pred_acc_percents = CV.cross_validation(ME.ridge_regression, orig_y, phi[:split_coord,:], k_folds, lambda_)
print("This is average loss: ", avg_loss)
print("This is average error in prediction: ", avg_preds)

#finding the weights
w, loss=ME.ridge_regression(orig_y,phi[:split_coord,:],lambda_)

#predicting labels
y_predicted=P1H.predict_labels(w,phi[split_coord:,:])

#creating the csv file
name="step_by_step_run.csv"
create_csv_submission(pred_ids, y_predicted, name)