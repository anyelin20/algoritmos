# Importe de Librerías
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression

# Definimos las rutas para cargar los datasets
mat = pd.read_csv('datasets/student-mat.csv', sep=';')
por = pd.read_csv('datasets/student-por.csv', sep=';')

# Definimos las columnas
num_cols = ['age','Medu','Fedu','traveltime','studytime','failures',
            'famrel','freetime','goout','Dalc','Walc','health','absences','G1','G2']
cat_cols = [c for c in mat.columns if c not in num_cols + ['G3']]

# Aplicamos la función del Benchmarking
def benchmark(df, name):
    X, y = df.drop('G3', axis=1), df['G3']

    # Implementamos la Búsqueda Exhaustiva
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer()),
        ('scaler', StandardScaler()),
        ('poly', PolynomialFeatures(include_bias=False))
    ])
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer([('num', num_pipeline, num_cols),
                                      ('cat', cat_pipeline, cat_cols)])
    pipe = Pipeline([('preproc', preprocessor), ('reg', LinearRegression())])
    param_grid = {
        'preproc__num__imputer__strategy': ['mean','median'],
        'preproc__num__scaler': [StandardScaler(), MinMaxScaler()],
        'preproc__num__poly__degree': [1,2]
    }
    # Medimos el tiempo que tarda la Búsqueda Exhaustiva en cada Dataset
    start = time.time()

    gs = GridSearchCV(pipe, param_grid, cv=5, scoring='r2', n_jobs=-1)
    gs.fit(X, y)
    grid_time = time.time() - start
    
    # Implementamos Algoritmo Genético 
    def make_model(g):
        s = 'mean' if g[0]==0 else 'median'
        sc = StandardScaler() if g[1]==0 else MinMaxScaler()
        d = 1 if g[2]==0 else 2
        num_p = Pipeline([('imputer', SimpleImputer(strategy=s)),
                          ('scaler', sc),
                          ('poly', PolynomialFeatures(degree=d, include_bias=False))])
        pre = ColumnTransformer([('num', num_p, num_cols),
                                 ('cat', cat_pipeline, cat_cols)])
        return Pipeline([('preproc', pre), ('reg', LinearRegression())])
    def fitness(gene):
        return cross_val_score(make_model(gene), X, y, cv=5, scoring='r2', n_jobs=-1).mean()
    
    # Evaluamos el tiempo que tarda el Algoritmo Genético en cada Dataset
    start = time.time()


    genes = np.array([[i,j,k] for i in [0,1] for j in [0,1] for k in [0,1]])
    scores = [fitness(g) for g in genes]
    best_idx = int(np.argmax(scores))
    ga_time = time.time() - start
    
    return {
        'dataset': name,
        'method': 'Grid Search',
        'R2': gs.best_score_,
        'time_s': grid_time
    }, {
        'dataset': name,
        'method': 'Genetic(8)',
        'R2': scores[best_idx],
        'time_s': ga_time
    }

# Ejecutamos el benchmarking
results = []
for df, name in [(mat,'Matemáticas'), (por,'Portugués')]:
    g, c = benchmark(df, name)
    results.extend([g, c])

df_results = pd.DataFrame(results)

# Observamos los resultados
print("Benchmarking de la implementación de los algoritmos en cada Dataset: ")
print(df_results.to_markdown(index=False))