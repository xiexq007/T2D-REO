# T2D-REOs

## What

- This is a novel data analysis approach based on the within-sample relative expression orderings of genes (REOs) and machine learning to characterize the transcriptomic differences between T2D and the healthy pancreatic islets.
- The proposed method can efficiently integrate data from divergent gene expression data sets and help discover some potential disease-specific signatures.
- Biomarkers (reverse REOs) found using this method have excellent predictive performance in predicting the occurrence of T2D on both training **(AUC = 0.886)** and independent testing data **(AUC = 0.839)**.
- These signatures can help elucidate the underlying pathogenic mechanisms of T2D.

## How

#### Datasets
- Three divergent bulk pancreatic islets expression profiles (GSE76894, GSE164416, GSE54279).
#### Method
1.  The overlapped reverse REOs were identified from three islets traning data subsets.
2.  **MIC, ANOVA, and mRMR** were applied to get the rankings of overlapped reverse REOs.
4.  Four classification algorithms-based models were constructed and compared to select the optimal model **(SVM-based model)**.
5.  Further **feature screening and parameter tuning** were implemented, the **top 7 REOs** selected by **mRMR** were the optimal feature subset.
6.  Based on **7 reverse REOs and SVM**, the model with excellent predictive power was constructed.

