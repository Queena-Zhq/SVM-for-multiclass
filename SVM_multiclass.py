##################################################################################
# !/usr/bin/env python3
# _*_ coding:utf-8 _*_
# Author: Qi Zhang
# Date: 30th Oct 2019
# Description: This code is about SVM in multi-class classification
#              Two strategies have been used, one-vs-rest and one-vs-one
#              The data sets are digits with 10 classes and wine with 3 classes
#              Grid search with 5-fold is used for optimal hyper-parameters
#              20*5-fold cross-validation is used for evaluate strategies in accuracy and time(training and test)
#              Running all strategies and data sets cost around 30m
##################################################################################

import time
from sklearn.svm import SVC
from sklearn.datasets import load_digits, load_wine
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split, KFold
import numpy as np

# This function uses grid search for optimal hyper-parameters
# C is set 7 possible values and gamma is the same
# 4 kernels are all run for the best performance
# data set is divided into 5 parts, 4 parts are used for training and one part are used for test
# For the best kernel, we test each optimal hyper-parameters for every kernel, and select the one with best result
def Best_param(x, y, strategy):
    kerlist = ["linear", "poly", "rbf", "sigmoid"]
    score_res = 0
    ker_res = ""
    C1 = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
    gamma = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
    best_CG = {}
    ans = []
    accu_rate = 0
    parameters = {"estimator__C": C1, "estimator__gamma": gamma}
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    for i in kerlist:
        # specify the condition, strategy is one-vs-rest
        if strategy == "OVR":
            model = OneVsRestClassifier(SVC(kernel=i))
            model_tunning = GridSearchCV(model, parameters, cv=5)
            clf = model_tunning.fit(x_train, y_train)
            accu_rate = model_tunning.score(x_test, y_test)
            print("The kernel is : " + i + " and the best param of OVR is : " + str(model_tunning.best_params_))
            print("the accuracy rate of test data is : " + str(accu_rate))
            if accu_rate > score_res:
                score_res = accu_rate
                ker_res = i
                best_CG = model_tunning.best_params_
        # specify the condition, strategy is one-vs-one
        else:
            model = OneVsOneClassifier(SVC(kernel=i))
            model_tunning = GridSearchCV(model, parameters, cv=5)
            clf = model_tunning.fit(x_train, y_train)
            accu_rate = model_tunning.score(x_test, y_test)
            print("The kernel is : " + i + "the best param of OVO is :" + str(model_tunning.best_params_))
            print("the accuracy rate of test data is : " + str(accu_rate))
            if accu_rate > score_res:
                score_res = accu_rate
                ker_res = i
                best_CG = model_tunning.best_params_
    ans.append(ker_res)
    ans.append(best_CG["estimator__C"])
    ans.append(best_CG["estimator__gamma"])
    print("Best parameters is : " + str(ans))
    return ans

# This function is 20 iterations with 5-fold cross-validation
# cross-validation score for each time is recorded
# each time choose one of 5-fold split to calculate training time and test time
def iteration_(x, y, strategy, param):

    result = []
    train_time = 0
    test_time = 0
    n_folds = 5
    my_gamma = float(param.pop())
    my_C = float(param.pop())
    my_kernel = param.pop()

    #20 runs for 5 cross_validation in both OVR and OVO
    for i in range(0, 20):
        # specify the condition, strategy is one-vs-rest
        if strategy == "OVR":
            print("This is OVR No : " + str(i) + " Cross_Validation")
            model = OneVsRestClassifier(SVC(kernel=my_kernel,C = my_C,gamma=my_gamma))
        # specify the condition, strategy is one-vs-one
        else:
            print("This is OVO No : " + str(i) + " Cross_Validation")
            model = OneVsOneClassifier(SVC(kernel=my_kernel,C = my_C,gamma=my_gamma))

        # cross-validation for one iteration and record the mean score
        kf = KFold(n_splits=n_folds, shuffle=True)
        scores = cross_val_score(model, x, y, cv=kf)
        score = scores.mean()
        result.append(score)

        # just record once for training time and test time in one iteration
        for train_index,test_index in kf.split(x,y):
            x_train,x_test = np.array(x)[train_index],np.array(x)[test_index]
            y_train,y_test = np.array(y)[train_index],np.array(y)[train_index]
            break

        train_start = time.time()
        clf = model.fit(x_train,y_train)
        train_end = time.time()

        test_start = time.time()
        model.predict(x_test)
        test_end = time.time()
        train_time = train_time + (train_end - train_start)
        test_time = test_time + (test_end - test_start)

        print("The mean socre is : " + str(score))
        print("The training time is : " + str(train_end - train_start))
        print("The test time is : " + str(test_end - test_start))

    print("This is the analyse data : \n")
    print("Total training time is : " + str(train_time))
    print("Total test time is : " + str(test_time))
    print("median accuracy rate is : " + str(np.median(sorted(result))))
    print("mean accuracy rate is : " + str(np.mean(sorted(result))))

#
#
# main function
if __name__ == '__main__':
    best_p_OVR1 = []
    best_p_OVR2 = []
    best_p_OVO1 = []
    best_p_OVO2 = []

    digits = load_digits()
    wine = load_wine()

    x, y = digits.data, digits.target
    y1 = label_binarize(y, classes=list(range(10)))

    k, z = wine.data, wine.target
    z1 = label_binarize(z, classes=list(range(3)))

    # data set is digits with 64 features and 10 classes
    # get best parameters for One-Vs-Rest and training 20 runs * 5 fold cross validation
    best_p_OVR1 = Best_param(x, y1, "OVR")
    iteration_(x, y1, "OVR", best_p_OVR1)
    # # get best parameters for One-VS-One and training 20 runs * 5 fold cross validation
    best_p_OVO1 = Best_param(x, y, "OVO")
    iteration_(x, y, "OVO", best_p_OVO1)

    print("=======================================================================\n\n\n\n\n")

    # data set is wine with 13 features and 3 classes
    # get best parameters for One-VS-Rest and training 20 runs * 5 fold cross validation
    best_p_OVR2 = Best_param(k, z1, "OVR")
    iteration_(k, z1, "OVR", best_p_OVR2)
    # get best parameters for One-VS-One and training 20 runs * 5 fold cross validation
    best_p_OVO2 = Best_param(k, z, "OVO")
    iteration_(k, z, "OVO", best_p_OVO2)
