import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score 
from joblib import dump


# model construction
def svm(x_train, y_train, feature_subset):
    
    x_np = np.array(x_train.loc[:, feature_subset])
    y_np = np.array(y_train)
    
    kf = KFold(n_splits=3,shuffle=True,random_state=42)
    svm_model = SVC(C=0.03125,kernel="rbf",gamma=0.03125,probability=True,random_state=42)

    true_labels = []
    pre_labels = []
    pre_probas = []
    score_cv_auc_list = []

    for train_index, test_index in kf.split(x_np):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = x_np[train_index], x_np[test_index]
        Y_train, Y_test = y_np[train_index], y_np[test_index]
        svm_model.fit(X_train,Y_train) 
        pre_label = svm_model.predict(X_test) 
        pre_labels = np.concatenate([pre_labels,pre_label])  
  
        pre_proba = svm_model.predict_proba(X_test)[:,1]
        pre_probas = np.concatenate([pre_probas,pre_proba])
        true_labels = np.concatenate([true_labels,Y_test])
        score_cv_auc_list.append(roc_auc_score(Y_test,pre_proba))
    
    print(roc_auc_score(true_labels, pre_probas))
    print(score_cv_auc_list,np.mean(score_cv_auc_list))
    dump(svm_model, 'svm_model.joblib')
    return svm_model