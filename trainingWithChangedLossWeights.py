##
import helper_multitask

## train model with age classifier
LARGE_VERSION = [True]
ALPHAS = [0.25]
DROPOUTS = [0.7]
L2 = [True, False]

for large in LARGE_VERSION:
    for alpha in ALPHAS:
        for dropout in DROPOUTS:
            for l2 in L2:
                helper_multitask.change_loss_function_while_training("classification", "images/featureTableTrain.csv",
                                                                     "images/featureTableVal.csv",
                                                                     alpha=alpha,
                                                                     dropout=dropout,
                                                                     epochs_face=10,
                                                                     epochs_mask=10,
                                                                     epochs_age=50,
                                                                     large_version=large,
                                                                     regularizer=l2)

## train model with age regressor
DROPOUTS = [0.2, 0.5, 0.7]
LARGE_VERSION = [False, True]
ALPHAS = [0.25]
L2 = [True, False]
for large in LARGE_VERSION:
    for alpha in ALPHAS:
        for dropout in DROPOUTS:
            for l2 in L2:
                helper_multitask.change_loss_function_while_training("regression", "images/featureTableTrain.csv",
                                                                     "images/featureTableVal.csv",
                                                                     alpha=alpha,
                                                                     dropout=dropout,
                                                                     epochs_face=10,
                                                                     epochs_mask=10,
                                                                     epochs_age=50,
                                                                     large_version=large,
                                                                     regularizer=l2)
