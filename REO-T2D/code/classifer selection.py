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
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    model = SVC(probability=True, random_state=42)
    
    true_labels = []
    pre_labels = []
    pre_probas = []
    
    for train_index, test_index in kf.split(x_train_np):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = x_train_np[train_index], x_train_np[test_index]
        Y_train, Y_test = y_train_np[train_index], y_train_np[test_index]
        model.fit(X_train, Y_train) 
        pre_label = model.predict(X_test)
        pre_labels = np.concatenate([pre_labels, pre_label])  
        pre_proba = model.predict_proba(X_test)[:, 1]
        pre_probas = np.concatenate([pre_probas, pre_proba])
        true_labels = np.concatenate([true_labels, Y_test])
    
    svm = pd.DataFrame({"ture_label": true_labels, "pre_probas": pre_probas})
    return svm

# RF
def model_rf(x_train, y_train):
    x_train_np = np.array(x_train) 
    y_train_np = np.array(y_train)
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    model = RandomForestClassifier(random_state=42)
    
    true_labels = []
    pre_labels = []
    pre_probas = []

    for train_index, test_index in kf.split(x_train_np):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = x_train_np[train_index], x_train_np[test_index]
        Y_train, Y_test = y_train_np[train_index], y_train_np[test_index]
        model.fit(X_train, Y_train) 
        pre_label = model.predict(X_test)
        pre_labels = np.concatenate([pre_labels, pre_label])   
        pre_proba = model.predict_proba(X_test)[:, 1]
        pre_probas = np.concatenate([pre_probas, pre_proba])
        true_labels = np.concatenate([true_labels, Y_test])
        
    rf = pd.DataFrame({"ture_label": true_labels, "pre_probas": pre_probas})
    return rf  

# LR
def model_lr(x_train, y_train):
    x_train_np = np.array(x_train)  
    y_train_np = np.array(y_train)
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    model = LogisticRegression(random_state=42)
    
    true_labels = []
    pre_labels = []
    pre_probas = []
    
    for train_index, test_index in kf.split(x_train_np):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = x_train_np[train_index], x_train_np[test_index]
        Y_train, Y_test = y_train_np[train_index], y_train_np[test_index]
        model.fit(X_train, Y_train) 
        pre_label = model.predict(X_test)
        pre_labels = np.concatenate([pre_labels, pre_label])  
        pre_proba = model.predict_proba(X_test)[:, 1]
        pre_probas = np.concatenate([pre_probas, pre_proba])
        true_labels = np.concatenate([true_labels, Y_test])
        
    lr = pd.DataFrame({"ture_label": true_labels, "pre_probas": pre_probas})
    return lr 

# XGBoost
def model_xgb(x_train, y_train):
    x_train_np = np.array(x_train)  
    y_train_np = np.array(y_train)
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    model = XGBClassifier(random_state=42)
    
    true_labels = []
    pre_labels = []
    pre_probas = []
    
    for train_index, test_index in kf.split(x_train_np):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = x_train_np[train_index], x_train_np[test_index]
        Y_train, Y_test = y_train_np[train_index], y_train_np[test_index]
        model.fit(X_train, Y_train) 
        pre_label = model.predict(X_test)
        pre_labels = np.concatenate([pre_labels, pre_label])  
        pre_proba = model.predict_proba(X_test)[:, 1]
        pre_probas = np.concatenate([pre_probas, pre_proba])
        true_labels = np.concatenate([true_labels, Y_test])
       
    xgb = pd.DataFrame({"ture_label": true_labels, "pre_probas": pre_probas})
    return xgb 

