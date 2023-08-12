import pandas as pd
import numpy as np

# getting stable gene pairs in T2D and ND samples respectively

def t2d_matrix(x):
    sample_num = t2d_exp.shape[1]
    diff = x - t2d_exp
    diff_po = diff > 0
    diff_ne = diff < 0
    diff_po_num = diff_po.sum(axis=1)  
    diff_ne_num = diff_ne.sum(axis=1)
    # set the threshold for obtaining REOs to 65%
    diff_po_re = (diff_po_num > (0.65 * sample_num)) * 1
    diff_ne_re = (diff_ne_num > (0.65 * sample_num)) * -1
    return (diff_po_re + diff_ne_re)

def nd_matrix(x):
    sample_num = nd_exp.shape[1]
    diff = x - nd_exp
    diff_po = diff > 0
    diff_ne = diff < 0
    diff_po_num = diff_po.sum(axis=1)  
    diff_ne_num = diff_ne.sum(axis=1)
    # set the threshold for obtaining REOs to 65%
    diff_po_re = (diff_po_num > (0.65 * sample_num)) * 1
    diff_ne_re = (diff_ne_num > (0.65 * sample_num)) * -1
    return (diff_po_re + diff_ne_re)

t2d_REOs = t2d_exp.apply(t2d_matrix,axis=1)
nd_REOs = nd_exp.apply(nd_matrix,axis=1)

# getting reveral stable gene pairs

def revese_REOs(t2d_REOs, nd_REOs):
    t2d_REOs[t2d_REOs==0] = 100  
    nd_REOs[nd_REOs==0] = 100  
    reverse_REOs = t2d_REOs + nd_REOs
    reverse_REOs_index = np.argwhere(abs(reverse_REOs).values==0)
    reverse_REOs_index = reverse_REOs_index.tolist() 
    uni_reverse_REOs_index = []
    for g in reverse_REOs_index:
        if g[0] < g[1]:   
            uni_reverse_REOs_index.append(g)
    
    symbol = reverse_REOs.index.values.tolist()  
    g1 = []
    g2 = []
    for i in uni_reverse_REOs_index:
        g1.append(symbol[i[0]])
        g2.append(symbol[i[1]])
        reverse_gene_pair = pd.DataFrame({"g1":g1, "g2":g2})
    return reverse_gene_pair

