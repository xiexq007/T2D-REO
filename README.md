# T2D-REO
### What
1. This is a intelligent model using the within-sample relative expression ordering of gene (REO) to predict the risk of type 2 diabetes (T2D).
2. The proposed model has an encouraging predictive power on both training **(AUC = 0.981)** and independent testing data **(AUC = 0.847)**.

### How
#### Datasets
Bulk pancreas islet expression profiles
#### Method

- The reversal gene pairs were first identified from three traning data subsets based on REOs strategy.
- Then we obtained the overlapped reversal gene pairs from the extracted gene pairs.
- Feature selection methods (MIC, ANOCA, mRMR) were applied to get the ranking of overlapped gene pairs in three training data subsets.
- By weighting its ranking in three training data subsets, we got the ranking of each reverse gene pair in the whole training dataset.
- We constructed and compared four common classification algorithms-based models to select the optimal model.
- To optimize the performance of the selected optimal model, we implemented feature screening and parameter tuning.
- The top 15 gene pairs selected by mRMR method were the optimal feature subset.
- And based on these 15 gene pairs, we constructed a model which could accurately differentiate diabetic and non-diabetic populations.

### Usage

