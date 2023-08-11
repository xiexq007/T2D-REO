#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score 

# IFS
def svm_ifs(x_train, y_train, order_list):
    
    selected_feature = [] 
    model_auc_list = []
    
    svm_model = SVC (probability=True, random_state=5)
    kf = KFold(n_splits=5, shuffle=True, random_state=5)
    
    for feature in order_list:
    selected_feature.append(feature)
    newx_np = np.array(x_train.loc[:, selected_feature]) 
    y_np = np.array(y_train)
    true_labels = []
    pre_labels = []
    pre_probas = []
   
    for train_index, test_index in kf.split(newx_np):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = newx_np[train_index], newx_np[test_index]
        Y_train, Y_test = y_np[train_index], y_np[test_index]
        svm_model.fit(X_train, Y_train) 
        pre_label = svm_model.predict(X_test) 
        pre_labels = np.concatenate([pre_labels, pre_label])  
        pre_proba = svm_model.predict_proba(X_test)[:, 1]
        pre_probas = np.concatenate([pre_probas, pre_proba])
        true_labels = np.concatenate([true_labels, y_test])
        
    model_auc = roc_auc_score(true_labels,pre_probas)  
    model_auc_list.append(model_auc) 
    print("Epoch:",selected_feature.index(feature))
    print("The selected features:",selected_feature)
    print("The AUC is:",model_auc)
    result = pd.DataFrame({"Feature_num": range(0,len(order_list),1), "5_CV_AUC": model_auc_list})
    return result

