# T2D-REO
### What
1. This is a intelligent model using the within-sample relative expression ordering of gene (REO) to predict the risk of type 2 diabetes (T2D).
2. The proposed model has an encouraging predictive power on both training **(AUC = 0.981)** and independent testing data **(AUC = 0.847)**.

### How
#### Datasets
Bulk pancreas islet expression profiles
#### Method

- The reversal gene pairs were first identified from three traning data subsets based on REO strategy.
- Then we obtained the overlapped reversal gene pairs from the extracted gene pairs.
- Feature selection methods (MIC, ANOCA, mRMR) were applied to get the ranking of overlapped gene pairs in three training data subsets.
- By weighting its ranking in three training data subsets, we got the ranking of each reverse gene pair in the whole training dataset.
- We constructed and compared four common classification algorithms-based models to select the optimal model.
- To optimize the performance of the selected optimal SVM-based model, we implemented feature screening and parameter tuning.
- The top 15 gene pairs selected by mRMR method were the optimal feature subset.
- And based on these 15 gene pairs and SVM, we constructed a model that can accurately differentiate diabetic and non-diabetic populations.

### Usage

- We saved the trained model in the file model/svm_model.joblib for easy access by the readers. 
- You can use the provided example file for making predictions on sample instances.
- For predictions of your own instance data, please process the data in the format of example_x (containing 15 key gene pairs) using the provided 'feature encoding' code

#### Example
```
from joblib import load
model = load('model/svm_model.joblib')
example_y_pred = model.predict(example_x)
```



