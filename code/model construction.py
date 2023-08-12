import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score 
from sklearn.metrics import roc_auc_score
from joblib import dump

# parameter tuning
def para_tuning(x_train, y_train):
    
    # Set the parameter search range
    param_grid = {     
    "kernel": [ "rbf"] ,     
    "C": np.logspace(-5, 13, num=10, base=2), 
    "gamma": np.logspace(-13, -5, num=10, base=2)}
    
    svm_grid = GridSearchCV(estimator=SVC(random_state=5, probability=True), param_grid=param_grid, scoring="roc_auc", cv=5)
    svm_grid.fit(x_train, y_train)
    parameter = svm_grid.best_params_
    return parameter

# model construction
def svm(x_train, y_train, feature_subset):
    new_x = x_train.loc[:, feature_subset]
    newx_np = np.array(new_x)  
    y_np = np.array(y)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=5)
    # using the optimal parameters after parameter tuning
    svm_model = SVC(C=8.0, kernel="rbf", gamma=0.0026577929690905803, probability=True, random_state=5)
    true_labels = []
    pre_labels = []
    pre_probas = []
    score_cv_acc_list = []
    score_cv_auc_list = []
    
    for train_index, test_index in kf.split(newx_np):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = newx_np[train_index], newx_np[test_index]
        Y_train, Y_test = y_np[train_index], y_np[test_index]
        svm_model.fit(X_train,Y_train) 
        pre_label = svm_model.predict(X_test) 
        pre_labels = np.concatenate([pre_labels, pre_label])  
        score_cv_acc_list.append(accuracy_score(Y_test, pre_label))
        pre_proba = svm_model.predict_proba(X_test)[:, 1]
        pre_probas = np.concatenate([pre_probas, pre_proba])
        true_labels = np.concatenate([true_labels, Y_test])
        score_cv_auc_list.append(roc_auc_score(Y_test, pre_proba))
        
        print(pre_labels)
        print(pre_probas)
        print(score_cv_acc_list, np.mean(score_cv_acc_list))
        print(score_cv_auc_list, np.mean(score_cv_auc_list))
        
    dump(svm_model, 'svm_model.joblib')
    return svm_model

