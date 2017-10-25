# standard libraries
#%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
#%load_ext autoreload
#%autoreload 2

# own functions
import proj1_helpers as P1H
import dataprocessing as DP
import methods as ME
import cross_validation as CV

#constants
train_path = 'train.csv'
test_path = 'test.csv'



##### IMPORTING DATA #####
orig_y, orig_x, orig_ids = P1H.load_csv_data(train_path, sub_sample=False) #remember to switch of subsample when running it "for real"
pred_y, pred_x, pred_ids = P1H.load_csv_data(test_path, sub_sample=False)
print(orig_x)



##### CLEANING DATA #####
# To provide clarity to which data I am processing
x = np.copy(orig_x)

# Cleans values that are -999 to zero, mean and median
no_clean = np.copy(orig_x)
clean_zero = DP.clean_data(x)
clean_mean = DP.clean_data(x, replace_no_measure_with_mean=True)
clean_medi = DP.clean_data(x, replace_no_measure_with_median=True)

# Make array to test for later
cleanDataArray = [no_clean, clean_zero, clean_mean, clean_medi]



##### MANIPULATION DATA WITH PCA AND POLYNOMIALS #####
# could make into a function that takes in different highest degree and number of PCA dimensions

degrees = (3, 5, 7, 9, 11, 13, 15, 20)
pc_keeps = (27, 28, 29, 30)

# So one can append new dataset to
clean_pc_poly = np.zeros((len(cleanDataArray), len(pc_keeps), len(degrees)), dtype=object)

# make phi for one and one cleaned dataset
for i, dataset in enumerate(cleanDataArray):

    # Doing PCA for different cleanings and reducing dimensionality too pc_keeps[j]
    for j, keep in enumerate(pc_keeps):

        pc = DP.pca(dataset, keep)[0]

        # making new polynomial feature for the given dataset
        for k, degree in enumerate(degrees):
            phi = DP.build_poly(pc, degree)
            clean_pc_poly[i, j, k] = phi[:, :]

# Making an matrix that are going to have
# rows = different methods to clean dataset
# columns = different degrees
print(clean_pc_poly.shape)


##### RIDGEREGR: TESTING DIFFERENT PHIS WITH DIFFERENT LAMDBAS #####
y = np.copy(orig_y)
y_log = DP.classify(y)

k_folds = 5
#parameters for ridge regression
lambdas=np.logspace(-11,1,10)
#parameters for logistic regression
max_iters = 5
gamma = 0.01

phis = np.copy(clean_pc_poly)
avg_predictions_of_phis = np.zeros((len(lambdas), phis.shape[0], phis.shape[1], phis.shape[2]), dtype=object)

for l, lambda_ in enumerate(lambdas):
    for i in range(phis.shape[0]):
        for j in range(phis.shape[1]):
            for k in range(phis.shape[2]):
                phi = phis[i,j,k]
                initial_w = np.zeros(((phi.shape[1], 1)))
                avg_loss, losses, avg_preds, pred_acc_percents = CV.cross_validation(ME.ridge_regression, y, phi, k_folds, lambda_)
                #avg_loss, losses, avg_preds, pred_acc_percents = CV.cross_validation(ME.logistic_regression, y_log, phi, k_folds, initial_w, max_iters, gamma)
                avg_predictions_of_phis[l,i,j,k] = avg_preds


# should become a matrix with containing average predicitions of different data and polynomials
print(avg_predictions_of_phis.shape)
#plt.plot(degrees,avg_predictions_of_phis[0,:])



##### PLOTS #####
# plot for clean_mean
# use the lambda you want to plot lambdas=np.logspace(-8,0,10)
l = 2
print('lamdba: ', lambdas[l])
#type of data cleaned
c = 2
print('clean_Strategy:', c)
print('0=noClean, 1=cleanZero, 2=cleanMean, 3=cleanMedian')
# pc_keeps = (15,18,21,24,27,30)
plt.plot(degrees,avg_predictions_of_phis[l,c,0,:],color='b', marker='*', label="pca 15")
plt.plot(degrees,avg_predictions_of_phis[l,c,1,:],color='r', marker='*', label="pca 18")
plt.plot(degrees,avg_predictions_of_phis[l,c,2,:],color='g', marker='*', label="pca 21")
plt.plot(degrees,avg_predictions_of_phis[l,c,3,:],color='m', marker='*', label="pca 24")
plt.plot(degrees,avg_predictions_of_phis[l,c,4,:],color='y', marker='*', label="pca 27")
plt.plot(degrees,avg_predictions_of_phis[l,c,5,:],color='k', marker='*', label="pca 30")

leg = plt.legend(loc=1, shadow=True)
axes = plt.gca()
axes.set_ylim([0.15,0.40])
plt.show
print(np.min(avg_predictions_of_phis[l,c,:,:]))
print(np.min(avg_predictions_of_phis))

