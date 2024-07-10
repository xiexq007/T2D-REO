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

## Raw data
#### Raw files of data analyzed in this manuscript were originally retrieved from:
1. GEO accession number [GSE76894](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE76894) (Solimena et al., 2018)
2. GEO accession number [GSE164416](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE164416) (Wigger et al., 2021)
3. GEO accession number [GSE54279](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE54279) (Krus et al., 2014)
4. GEO accession number [GSE86468](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE86468) (Lawlor et al., 2017)
5. GEO accession number [GSE118139](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE118139) (Wang et al., 2019)
6. GEO accession number [GSE184050](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE184050) (Chen et al., 2022)
7. GEO accession number [GSE78721](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE78721) (Tiwari et al., 2018)
8. GEO accession number [GSE153855](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE153855) (Ngara & Wierup, 2022)
