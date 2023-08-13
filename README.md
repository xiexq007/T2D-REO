# T2D-REO

## What

- This is a intelligent model using the within-sample relative expression ordering of gene (REO) to predict the risk of type 2 diabetes (T2D).
- The proposed model has an encouraging predictive power on both training **(AUC = 0.981)** and independent testing data **(AUC = 0.847)**.

## How

#### Datasets
- Bulk pancreas islet expression profiles (GSE76895, GSE41762, GSE164416).
#### Method
1.  Reversal gene pairs were identified from three traning data subsets based on **REO** strategy, and obtained the overlapped gene pairs.
2.  **MIC, ANOVA, and mRMR** were applied to get the rankings of overlapped gene pairs in three training data subsets.
3.  By weighting its ranking in three training data subsets, we got the ranking of each gene pair in the whole training dataset.
4.  Four common classification algorithms-based models were constructed and compared to select the optimal model **(SVM-based model)**.
5.  Further **feature screening and parameter tuning** were implemented, the **top 15 gene pairs** selected by **mRMR** method were the optimal feature subset.
6.  Based on **15 gene pairs and SVM**, the final model was constructed.

## Usage

- The trained model was saved in the **model/svm_model.joblib**, please install the required modules: scikit-learn, joblib.
- You can use the provided example **(example/example_x)** for testing.
- For predictions of your own instance data, please process the data in the format of example_x (containing 15 key gene pairs) using the provided **code/feature encoding** code.

#### Example
```
from joblib import load
model = load('model/svm_model.joblib')
example_y_pred = model.predict(example_x)
```



