import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from algortimos import StudentPerformanceEvaluator  # Asegúrate de tener el archivo con esa clase

# 1. Cargar el dataset combinado
df = pd.read_csv("Seleccion_Algoritmos/student-combined.csv", sep=",")
df.columns = df.columns.str.strip()

# 2. Codificar columnas categóricas
categorical_features = [
    'school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob',
    'reason', 'guardian', 'schoolsup', 'famsup', 'paid', 'activities',
    'nursery', 'higher', 'internet', 'romantic'
]

le = LabelEncoder()
for col in categorical_features:
    if col in df.columns:
        df[col] = le.fit_transform(df[col])

# 3. Definir X y y
X = df.drop(columns=['G3'])  # Puedes también eliminar G1 y G2 si quieres más reto
y = df['G3']

# 4. Separar datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Evaluar
evaluator = StudentPerformanceEvaluator(X_train, X_test, y_train, y_test)

# Búsqueda genética
genetic_results = evaluator.genetic_search()

# Búsqueda exhaustiva
exhaustive_results = evaluator.exhaustive_search()

# Comparar resultados
comparison_df = evaluator.compare_results(genetic_results, exhaustive_results)
print(comparison_df)
