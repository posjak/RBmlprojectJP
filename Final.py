import pandas as pd
import numpy as np
import neptune.new as neptune
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
standardizer=StandardScaler()
np.random.seed(2014)

#neptune
run = neptune.init(project='posjak/DSprojektML',
                   api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJkZDFlOWQyOS02NDZmLTQxYTItODA1OC1kYmVlZGI4ODdhYmYifQ==')
def reduce_mem_usage(props):
    start_mem_usg = props.memory_usage().sum() / 1024 ** 2
    print("Memory usage of properties dataframe is :", start_mem_usg, " MB")
    NAlist = []  # Keeps track of columns that have missing values filled in.
    for col in props.columns:
        if props[col].dtype != object:  # Exclude strings

            # Print current column type
            # print("******************************")
            # print("Column: ",col)
            # print("dtype before: ",props[col].dtype)

            # make variables for Int, max and min
            IsInt = False
            mx = props[col].max()
            mn = props[col].min()

            # Integer does not support NA, therefore, NA needs to be filled
            if not np.isfinite(props[col]).all():
                NAlist.append(col)
                props[col].fillna(mn - 1, inplace=True)

                # test if column can be converted to an integer
            asint = props[col].fillna(0).astype(np.int64)
            result = (props[col] - asint)
            result = result.sum()
            if result > -0.01 and result < 0.01:
                IsInt = True

            # Make Integer/unsigned Integer datatypes
            if IsInt:
                if mn >= 0:
                    if mx < 255:
                        props[col] = props[col].astype(np.uint8)
                    elif mx < 65535:
                        props[col] = props[col].astype(np.uint16)
                    elif mx < 4294967295:
                        props[col] = props[col].astype(np.uint32)
                    else:
                        props[col] = props[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        props[col] = props[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        props[col] = props[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        props[col] = props[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        props[col] = props[col].astype(np.int64)

                        # Make float datatypes 32 bit
            else:
                props[col] = props[col].astype(np.float32)

            # Print new column type
            # print("dtype after: ",props[col].dtype)
            # print("******************************")

    # Print final result
    print("___MEMORY USAGE AFTER COMPLETION:___")
    mem_usg = props.memory_usage().sum() / 1024 ** 2
    print("Memory usage is: ", mem_usg, " MB")
    print("This is ", 100 * mem_usg / start_mem_usg, "% of the initial size")
    return props

#import
def load_data():
    props = pd.read_csv(os.path.join('data','train_data.csv'), header=None)
    X = reduce_mem_usage(props)
    props = pd.read_csv(os.path.join('data','train_labels.csv'), header=None)
    y = reduce_mem_usage(props)
    props = pd.read_csv(os.path.join('data','test_data.csv'), header=None)
    test = reduce_mem_usage(props)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify = y)
    y_train=np.ravel(y_train)
    return X, y, X_train, y_train, test

X, y, X_train, y_train, test = load_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify = y)

def confusion_matrix(classifier):
    m = classifier.fit(X, y)
    m.fit(X_train, y_train)
    y_pred = m.predict(X_test)
    print(confusion_matrix(y_test, y_pred))

def y_to_csv(y):
    y.to_csv()


def pipeline_model(x_1: np.array, x_2: np.array, y_1: np.array, y_2: np.array):
    pipe = Pipeline([('pca', PCA(n_components=0.99)), ('scaler', StandardScaler()), ('classifier', SVC())])

    search_space = [{'scaler': [StandardScaler()],
                     'pca': [PCA(n_components=0.99)]},
                    {'classifier': [SVC()],
                     'classifier__kernel': ['linear', 'poly'],
                     'classifier__class_weight': ['balanced'],
                     'classifier__C': np.logspace(1, 4, 5)},
                    {'classifier': [KNeighborsClassifier()],
                     'classifier__n_neighbors': [2, 4, 6, 8, 10],
                     'classifier__algorithm': ['auto']},
                    {'classifier': [GradientBoostingClassifier()],
                     'classifier__n_estimators': [50],
                     'classifier__max_depth': [3]}]

    grid_search = GridSearchCV(pipe,
                               search_space,
                               cv=3,
                               verbose=2,
                               n_jobs=-1,
                               scoring='f1')

    best_model = grid_search.fit(x_1, y_1)
    print(best_model.best_estimator_)
    y_pred = best_model.predict(x_2)
    print("The score of the model is:", f1_score(y_2, y_pred))
    print(confusion_matrix(y_2, y_pred))


pipeline_model(X_train, X_test, y_train, y_test)


def make_y_test(x_1: np.array, x_2: np.array, y_1: np.array):
    pipe = Pipeline([('pca', PCA(n_components=0.99)), ('scaler', StandardScaler()), ('classifier', SVC())])

    search_space = [{'scaler': [StandardScaler()],
                     'pca': [PCA(n_components=0.99)]},
                    {'classifier': [SVC()],
                     'classifier__kernel': ['linear', 'poly'],
                     'classifier__class_weight': ['balanced'],
                     'classifier__C': np.logspace(1, 4, 5)},
                    {'classifier': [KNeighborsClassifier()],
                     'classifier__n_neighbors': [2, 4, 6, 8, 10],
                     'classifier__algorithm': ['auto']},
                    {'classifier': [GradientBoostingClassifier()],
                     'classifier__n_estimators': [50],
                     'classifier__max_depth': [3]}]

    grid_search = GridSearchCV(pipe,
                               search_space,
                               cv=3,
                               verbose=2,
                               n_jobs=-1,
                               scoring='f1')

    best_model = grid_search.fit(x_1, y_1)
    y_pred = best_model.predict(x_2)

def y_to_csv(y_pred):
    return y_pred

make_y_test(X, test, y)


