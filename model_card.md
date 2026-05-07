# Model Card

Model developed as part of Udacity Machine Learning DevOps Engineer Nanodegree Programme: Course 4 - Deploying a scalable ML pipeline in production.

## Model Details
Random forest classification model with default hyperparameters in scikit-learn 1.7.2. 

## Intended Use
Model should be used to predict is a persons income is <=50k or not.

## Training Data
Publicly available Census Bureau data sourced from the UCI Machine Learning Repository. More info available [here](https://archive.ics.uci.edu/dataset/20/census+income). To use data for training a One Hot Encoder was used on categorical features and a Label Binarizer was used on the labels

## Evaluation Data
The original data had 32561 rows, and a 80-20 split was used to break this into training and test sets. No stratification was done.

## Metrics
Model was evaluated on slices of categorical features using precision, recall and $F_{\beta}$. See `slice_output.txt` for values. Sample for the the `sex` categorical feature.

| Category | Precision | Recall | $F_{\beta}$ |
|----------|-----------|--------|-------------|
| Male | 0.74 | 0.63 | 0.68 |
| Female | 0.71 | 0.59 | 0.65 |

## Ethical Considerations
We risk expressing the viewpoint that the attributes in this dataset are the only ones that are predictive of someone's income, even though we know this not to be the case.

## Caveats and Recommendations
This data was donated to the machine learning repository in 1996 and may no longer reflect real world conditions.