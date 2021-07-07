## EDA

1. Imported modules
2. reading data
3. split data into test and training data
4. missing data and duplicates
    - there is no problem with duplicates and missing values
5. descriptive statistics
    - the dataset is very diverse
    - dataset is very close to normal distribuution


## Metrics
1. Due to the lack of information on the data, we concluded that the most check if the metric would be f1_score.
2. To further forgive the difference between Precision and recall, we chose Average = 'weighted'.

## Models
1. At the beginning we checked many different options with different classifiers without any parameters. We got the best, that is:
    - GradientBoostingClassifier,
    - KNeighborsClassifie,
    - SVC
2. We checked the hyperopt, RandomizedSearchCV, GridSearchCV
3. Dummy Classifier returns f1-weighted of 0.91

## conclusion
- The best result we ever managed to get was GradientBoostingClassifier on the hyperopt, which reached 0.9975,
- You should learn github better, without it we wouldn't have problems with showing our results.


