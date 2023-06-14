import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# loading the diabetes dataset to a pandas DataFrame
df = pd.read_csv('./diabetes.csv')
X, Y = df.drop(columns='Outcome', axis=1), df['Outcome']

scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=42)

param_grid = {
    'C': [0.1, 1, 10, 100, 1000],
    'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
    'kernel': ['linear', 'rbf', 'sigmoid'],
}

grid = GridSearchCV(SVC(probability=True, cache_size=10000, class_weight='balanced'), param_grid, refit=True, verbose=4)
grid.fit(X_train, Y_train)

print(grid.best_params_)
print(grid.best_estimator_)

# accuracy score on the training data
X_train_prediction = grid.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy score of the training data : ', training_data_accuracy)

# accuracy score on the test data
X_test_prediction = grid.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy score of the test data : ', test_data_accuracy)

pickle.dump(grid, open('grid_model.sav', 'wb'))
