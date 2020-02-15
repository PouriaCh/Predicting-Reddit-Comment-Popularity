import json
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from methods import *
# Loading the dataset
current_folder = os.getcwd()
dataset_filename = "comments.json"
with open(os.path.join(current_folder, "data", dataset_filename)) as fp:
    data = json.load(fp)

# Splitting data into partitions

no_data_point = 12000
training_data_points = 10000
validation_data_points = 1000
test_data_points = 1000

train_data = data[ : training_data_points]
validation_data = data[training_data_points : (training_data_points + validation_data_points)]
test_data = data[(training_data_points + validation_data_points) : no_data_point]

print("Number of training data points: " + str(len(train_data)))
print("Number of evaluation data points: " + str(len(validation_data)))
print("Number of test data points: " + str(len(test_data)))

# Performance Evaluation on training/validation

no_text_features = 160
Eta0 = 1e-6
Beta0 = 0.0001
epsilon = 1e-6
X,Y,most_freq_words = Feature_Matrix(train_data, no_text_features)
W0 = weights_init(X.shape[1], True) # All_zero weights initialization if True

start1 = time.time()
W_LS = Least_Squares_Estimation(X, Y)
MSE_LS_training = Mean_Square_Error(X, Y, W_LS)
end1 = time.time()

print("MSE for LS on Training set is: " + str(MSE_LS_training) + " which took: " + str((end1-start1) * 1000) + " ms" + "")

start2 = time.time()
W_GD, MSE_GD_training = Gradient_Descent(X, Y, W0, Beta0, Eta0, epsilon)
end2 = time.time()
print("Final MSE for GD on Training set at " + str(len(MSE_GD_training)) + "'s epoch is :" + str(MSE_GD_training[-1]) +
     " which took: " + str((end2-start2) * 1000) + " ms" + "")


# Run on Validation dataset

X_validation, Y_validation, _ = Feature_Matrix(validation_data, no_text_features)

MSE_LS_validation = Mean_Square_Error(X_validation, Y_validation, W_LS)

print("MSE for LS on Validation set is: " + str(MSE_LS_validation))

MSE_GD_validation = Mean_Square_Error(X_validation, Y_validation, W_GD)

print("Final MSE for GD on Validation set is :" + str(MSE_GD_validation))

# Performance evaluation on test

X_test, Y_test,_ = Feature_Matrix(test_data, no_text_features)

MSE_LS_Test = Mean_Square_Error(X_test, Y_test, W_LS)

print("MSE for LS on Test set is: " + str(MSE_LS_Test))

MSE_GD_validation = Mean_Square_Error(X_test, Y_test, W_GD)

print("MSE for GD on Test set is :" + str(MSE_GD_validation))

# Plots
# training set
lineplot(range(MSE_GD_training.shape[0]), MSE_GD_training,
         x_label="epochs", y_label="MSE", title="Gradient Descent for Training set", gcolor='b')

