#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np

def encoding_matrix(exp, gene_pair):
    sample_num = exp.shape[1]
    gene_pair_num = gene_pair.shape[0] 
    gene_index = gene_pair.apply(lambda x: x[0]+"|"+x[1], axis=1)
    encoding_data = pd.DataFrame(index=exp.columns, columns=gene_index)

    for sample_name in exp.columns:
        for gene_name in gene_index:
            gene_name1, gene_name2 = gene_name.split("|")
            if exp.loc[gene_name1, sample_name] > exp.loc[gene_name2, sample_name]:
                encoding_data.loc[sample_name, gene_name] = 1
            else:
                encoding_data.loc[sample_name, gene_name] = -1
    return encoding_data

