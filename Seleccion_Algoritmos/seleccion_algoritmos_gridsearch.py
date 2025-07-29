
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Cargar dataset combinado
df = pd.read_csv("student-combined.csv")

# Codificación de variables categóricas
for col in df.select_dtypes(include=['object']).columns:
    df[col] = LabelEncoder().fit_transform(df[col])

# Variables predictoras y objetivo
X = df.drop(columns='G3')
y = pd.cut(df['G3'], bins=[-1, 9, 13, 20], labels=[0, 1, 2])

# División train-test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelos y parámetros
classifiers = {
    "LogisticRegression": (LogisticRegression(max_iter=1000), {
        "C": [0.1, 1, 10]
    }),
    "RandomForestClassifier": (RandomForestClassifier(), {
        "n_estimators": [50, 100],
        "max_depth": [5, 10]
    }),
    "SVC": (SVC(), {
        "C": [0.5, 1, 2],
        "kernel": ["linear", "rbf"]
    }),
    "KNeighborsClassifier": (KNeighborsClassifier(), {
        "n_neighbors": [3, 5, 7]
    })
}

# Benchmarking con Grid Search
results = []
for name, (clf, params) in classifiers.items():
    grid = GridSearchCV(clf, params, cv=5, scoring='accuracy')
    grid.fit(X_train, y_train)
    y_pred = grid.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results.append({
        "Algorithm": name,
        "Best Params": grid.best_params_,
        "Test Accuracy": acc
    })

# Mostrar resultados
df_results = pd.DataFrame(results)
print(df_results)
