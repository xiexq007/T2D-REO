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

def mic_order(data):
    name_list = []
    mic_list = []
    for gene_pair in data.columns:
    mic = mic_score(data[gene_pair],data["label"])
    name_list.append(gene_pair)
    mic_list.append(mic)
    mic = pd.DataFrame({'Gene_pair': name_list,'Mic': mic_list})
    mic = mic.sort_values(by="Mic", ascending=False)
    return mic

# getting weighted mic order in whole dataset
def merged_mic(mic_1, mic_2, mic_3):
    merged_mic = pd.merge(mic_1, mic_2, on='Gene_pair', how='left').merge(mic_3, on='Gene_pair', how='left')
    merged_mic['Order_sum'] = merged_mic[['Order_x', 'Order_y', 'Order']].sum(axis=1)
    merged_mic = merged_mic.sort_values(by="Order_sum")
    return merged_mic

# ANOVA
def anova_order(x_train, y_train):
    f_statistic, p_values = f_classif(x_train, y_train)
    anova = pd.DataFrame({"Gene_pair": x_train.columns, "F_value": f_statistic, "P_value": p_values})
    anova = anova.sort_values(by="P_value", ascending=True)  
    return anova

# getting weighted anova order in whole dataset
def merged_anova(anova_1, anova_2, anova_3):
    merged_anova= pd.merge(anova_1, anova_2, on='Gene_pair', how='left').merge(anova_3, on='Gene_pair', how='left')
    merged_anova['Order_sum'] = merged_anova[['Order_x', 'Order_y', 'Order']].sum(axis=1)
    merged_anova = merged_anova.sort_values(by="Order_sum")
    return merged_anova

# mRMR
def mRMR_order(x_train, y_train):
    data = pd.concat([x_train, y_train], axis=1)
    # feature_num is the number of features
    mRMR_order = pymrmr.mRMR(data, 'MIQ', feature_num)
    return mRMR_order

# getting weighted mrmr order in whole dataset
def merged_mrmr(mrmr_1, mrmr_2, mrmr_3):
    merged_mrmr= pd.merge(mrmr_1, mrmr_2, on='Gene_pair', how='left').merge(mrmr_3, on='Gene_pair', how='left')
    merged_mrmr['Order_sum'] = merged_mrmr[['Order_x', 'Order_y', 'Order']].sum(axis=1)
    merged_mrmr = merged_mrmr.sort_values(by="Order_sum")
    return merged_mrmr

