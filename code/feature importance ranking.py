import pandas as pd
import numpy as np
from minepy import MINE
from sklearn.feature_selection import f_classif
import pymrmr

# MIC
def mic_score(x_train, y_train):
    mine = MINE(alpha=0.6, c=15)
    mine.compute_score(x_train, y_train)
    return mine.mic()

def get_mic(x_train, y_train):
    
    name_list = []
    mic_list = []

    for gene_pair in x_train.columns:
        mic = mic_score(x_train[gene_pair],y_train)
        name_list.append(gene_pair)
        mic_list.append(mic)
    
    gene_mic = pd.DataFrame({'Feature':name_list,'Mic':mic_list})
    gene_mic = gene_mic.sort_values(by="Mic",ascending=False) 
    return gene_mic

# getting weighted mic order in whole dataset
def weighted_mic(mic_1, mic_2, mic_3):
    weighted_mic = pd.merge(mic_1, mic_2, on='Gene_pair', how='left').merge(mic_3, on='Gene_pair', how='left')
    weighted_mic['Order_sum'] = weighted_mic[['Order_x', 'Order_y', 'Order']].sum(axis=1)
    weighted_mic = weighted_mic.sort_values(by="Order_sum")
    return weighted_mic

# ANOVA
def get_anova(x_train, y_train):
    f_statistic, p_values = f_classif(x_train, y_train)
    anova = pd.DataFrame({"Gene_pair": x_train.columns, "F_value": f_statistic, "P_value": p_values})
    anova = anova.sort_values(by="P_value", ascending=True)  
    return anova

# getting weighted anova order in whole dataset
def weighted_anova(anova_1, anova_2, anova_3):
    weighted_anova= pd.merge(anova_1, anova_2, on='Gene_pair', how='left').merge(anova_3, on='Gene_pair', how='left')
    weighted_anova['Order_sum'] = weighted_anova[['Order_x', 'Order_y', 'Order']].sum(axis=1)
    weighted_anova = weighted_anova.sort_values(by="Order_sum")
    return weighted_anova

# mRMR
def get_mRMR(x_train, y_train):
    data = pd.concat([y_train, x_train], axis=1)
    # feature_num is the number of features
    mRMR_order = pymrmr.mRMR(data, 'MIQ', feature_num)
    return mRMR_order

# getting weighted mrmr order in whole dataset
def weighted_mrmr(mrmr_1, mrmr_2, mrmr_3):
    weighted_mrmr= pd.merge(mrmr_1, mrmr_2, on='Gene_pair', how='left').merge(mrmr_3, on='Gene_pair', how='left')
    weighted_mrmr['Order_sum'] = weighted_mrmr[['Order_x', 'Order_y', 'Order']].sum(axis=1)
    weighted_mrmr = weighted_mrmr.sort_values(by="Order_sum")
    return weighted_mrmr

