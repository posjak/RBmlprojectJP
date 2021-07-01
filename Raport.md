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
1. Przez brak dokładnych informacji na temat danych, doszliśmy do wniosku, że najabardziej odpowiednią metryką będzie f1_score.
2. Żeby jeszcze bardziej wywarzyć wyni między precision i recall, wybralismy average='weighted'.

## Models
1. Sprawdzaliśmy na początku wiele różnych opcji z różnymi klasyfikatorami bez żadnych parametrów. Wyciągneliśmy najlpesze czyli:
    - GradientBoostingClassifier,
    - KNeighborsClassifie,
    - SVC
2. Sprawdzaliśmy hyperopta, RandomizedSearchCV,GridSearchCV

## conclusion
- Najlepszy wynik jaki nam się w ogóle udało uzyskać to był GradientBoostingClassifier na hyperopcie, który osiągnął 0.9975,
- Należy zabierać się za projekt szybciej, tak jak wykładowca mówił,
- Należy lepiej opanować githuba, bez tego nie mielibyśmy problemów z pokazaniem naszych wyników.


