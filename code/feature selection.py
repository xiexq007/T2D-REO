import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score 

# IFS
def ifs(feature_list, x_train, y_train):
    selected_feature = [] 
    auc_list = []   
    hyperparameter_list = []  
    
    param_grid = {     
        "kernel": [ "rbf"] ,   
        "C": np.logspace(-5, 13, num=10, base=2), 
        "gamma": np.logspace(-13, -5, num=10, base=2)}
    kf = KFold(n_splits=3,shuffle=True,random_state=42)
    svm_grid = GridSearchCV(estimator=SVC(random_state=42,probability=True), param_grid=param_grid, scoring="roc_auc",cv=kf, n_jobs=-1)
    
    for feature in feature_list:
        selected_feature.append(feature)
        svm_grid.fit(x_train.loc[:, selected_feature], y_train)
        hyperparameter_list.append(svm_grid.best_params_)
        auc_list.append(svm_grid.best_score_)
        print("Epoch:",selected_feature.index(feature))
        print("The best parameter is:",svm_grid.best_params_)
        print("The 3 CV AUC is:",svm_grid.best_score_)
        
    ifs = pd.DataFrame({"Feature_num":range(1,12,1),"Parameter": hyperparameter_list, "AUC":auc_list})
    return ifs
