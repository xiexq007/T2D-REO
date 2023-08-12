import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score 

# SVM
def model_svm(x_train, y_train):
    x_train_np = np.array(x_train) 
    y_train_np = np.array(y_train)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=5)
    model_svm = SVC(probability=True, random_state=5)
    true_labels = []
    pre_labels = []
    pre_probas = []
    score_cv_auc_list = []
    
    for train_index, test_index in kf.split(x_train_np):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = x_train_np[train_index], x_train_np[test_index]
        Y_train, Y_test = y_train_np[train_index], y_train_np[test_index]
        model_svm.fit(X_train, Y_train) 
        pre_label = model_svm.predict(X_test)
        pre_labels = np.concatenate([pre_labels, pre_label])  
        pre_proba = model_svm.predict_proba(X_test)[:, 1]
        pre_probas = np.concatenate([pre_probas, pre_proba])
        true_labels = np.concatenate([true_labels, Y_test])
        score_cv_auc_list.append(roc_auc_score(Y_test, pre_proba))
        print(score_cv_auc_list, np.mean(score_cv_auc_list))
        svm = pd.DataFrame({"ture_label": true_labels, "pre_probas": pre_probas})
    return svm

# RF
def model_rf(x_train, y_train):
    x_train_np = np.array(x_train) 
    y_train_np = np.array(y_train)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=5)
    model_svm = RandomForestClassifier(random_state=5)
    true_labels = []
    pre_labels = []
    pre_probas = []
    score_cv_auc_list = []
    
    for train_index, test_index in kf.split(x_train_np):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = x_train_np[train_index], x_train_np[test_index]
        Y_train, Y_test = y_train_np[train_index], y_train_np[test_index]
        model_rf.fit(X_train, Y_train) 
        pre_label = model_svm.predict(X_test)
        pre_labels = np.concatenate([pre_labels, pre_label])   
        pre_proba = model_svm.predict_proba(X_test)[:, 1]
        pre_probas = np.concatenate([pre_probas, pre_proba])
        true_labels = np.concatenate([true_labels, Y_test])
        score_cv_auc_list.append(roc_auc_score(Y_test, pre_proba))
        print(score_cv_auc_list, np.mean(score_cv_auc_list))
        rf = pd.DataFrame({"ture_label": true_labels, "pre_probas": pre_probas})
    return rf  

# LR
def model_lr(x_train, y_train):
    x_train_np = np.array(x_train)  
    y_train_np = np.array(y_train)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=5)
    model_lr = LogisticRegression(random_state=5)
    true_labels = []
    pre_labels = []
    pre_probas = []
    score_cv_auc_list = []
    
    for train_index, test_index in kf.split(x_train_np):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = x_train_np[train_index], x_train_np[test_index]
        Y_train, Y_test = y_train_np[train_index], y_train_np[test_index]
        model_lr.fit(X_train, Y_train) 
        pre_label = model_lr.predict(X_test)
        pre_labels = np.concatenate([pre_labels, pre_label])  
        pre_proba = model_lr.predict_proba(X_test)[:, 1]
        pre_probas = np.concatenate([pre_probas, pre_proba])
        true_labels = np.concatenate([true_labels, Y_test])
        score_cv_auc_list.append(roc_auc_score(Y_test, pre_proba))
        print(score_cv_auc_list, np.mean(score_cv_auc_list))
        lr = pd.DataFrame({"ture_label": true_labels, "pre_probas": pre_probas})
    return lr 

# XGBoost
def model_lr(x_train, y_train):
    x_train_np = np.array(x_train)  
    y_train_np = np.array(y_train)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=5)
    model_xgb = XGBClassifier(random_state=5)
    true_labels = []
    pre_labels = []
    pre_probas = []
    score_cv_auc_list = []
    
    for train_index, test_index in kf.split(x_train_np):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = x_train_np[train_index], x_train_np[test_index]
        y_train, Y_test = y_train_np[train_index], y_train_np[test_index]
        model_xgb.fit(X_train, Y_train) 
        pre_label = model_xgb.predict(X_test)
        pre_labels = np.concatenate([pre_labels, pre_label])  
        pre_proba = model_xgb.predict_proba(X_test)[:, 1]
        pre_probas = np.concatenate([pre_probas, pre_proba])
        true_labels = np.concatenate([true_labels, Y_test])
        score_cv_auc_list.append(roc_auc_score(Y_test, pre_proba))
        print(score_cv_auc_list, np.mean(score_cv_auc_list))
        xgb = pd.DataFrame({"ture_label": true_labels, "pre_probas": pre_probas})
    return xgb 

