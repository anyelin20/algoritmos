# Importe de Librerías para el Código
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression


# Definimos las rutas para cargar los datasets
mat = pd.read_csv('datasets/student-mat.csv', sep=';')
por = pd.read_csv('datasets/student-por.csv', sep=';')

# Definimos las columnas de los conjuntos de datos
num_cols = [
    'age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures',
    'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health',
    'absences', 'G1', 'G2'
]
cat_cols = [col for col in mat.columns if col not in num_cols + ['G3']]


# ============== Implementación de Algoritmo de Búsqueda Exhaustiva ==============

# 1) Definimos la función para crear pipeline según genes
def make_pipeline(gene):
    strategy = 'mean' if gene[0] == 0 else 'median'
    scaler = StandardScaler() if gene[1] == 0 else MinMaxScaler()
    degree = 1 if gene[2] == 0 else 2
    
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy=strategy)),
        ('scaler', scaler),
        ('poly', PolynomialFeatures(degree=degree, include_bias=False))
    ])
    cat_pipeline = Pipeline([
        ('imp', SimpleImputer(strategy='most_frequent')),
        ('enc', OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer([
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols)
    ])
    return Pipeline([
        ('preproc', preprocessor),
        ('regressor', LinearRegression())
    ])

# 2) Hacemos una evaluación del fitness por CV
def fitness(gene, X, y):
    model = make_pipeline(gene)
    scores = cross_val_score(model, X, y, cv=5, scoring='r2', n_jobs=-1)
    return scores.mean()

# 3) Hacemos una busqueda genética manual con 3 genes y con 8 combinaciones
def genetic_search(df, name, pop_size=10, generations=10,
                   cx_prob=0.5, mut_prob=0.1, random_state=42):
    np.random.seed(random_state)
    X = df.drop('G3', axis=1)
    y = df['G3']
    
    # Inicializamos una población aleatoria 
    population = np.random.randint(0, 2, size=(pop_size, 3))
    
    best_gene, best_score = None, -np.inf
    for gen in range(1, generations + 1):
        # Evaluamos el fitness
        scores = np.array([fitness(g, X, y) for g in population])
        
        # Registramos la mejor de la generación
        idx_best = np.argmax(scores)
        if scores[idx_best] > best_score:
            best_score = scores[idx_best]
            best_gene = population[idx_best].copy()
        
        print(f"Gen {gen}: Mejor R² = {scores[idx_best]:.5f}")
        
        probs = (scores - scores.min()) + 1e-6
        probs /= probs.sum()
        
        # Creamos una nueva población
        new_pop = []
        while len(new_pop) < pop_size:
            parents = population[np.random.choice(pop_size, 2, p=probs)]
            if np.random.rand() < cx_prob:
                cx_point = np.random.randint(1, 3)  # punto de cruce 1 o 2
                child1 = np.concatenate([parents[0][:cx_point], parents[1][cx_point:]])
                child2 = np.concatenate([parents[1][:cx_point], parents[0][cx_point:]])
            else:
                child1, child2 = parents
            new_pop.extend([child1, child2])
        population = np.array(new_pop[:pop_size])
        

        mutation_mask = np.random.rand(pop_size, 3) < mut_prob
        population = np.where(mutation_mask, 1 - population, population)
    
    # Decodificamos el mejor gen
    imputer_str = 'mean' if best_gene[0] == 0 else 'median'
    scaler_name = 'StandardScaler' if best_gene[1] == 0 else 'MinMaxScaler'
    degree = 1 if best_gene[2] == 0 else 2
    print(f"\n{name} Dataset: Mejor configuración GA → "
          f"imputer={imputer_str}, scaler={scaler_name}, poly_degree={degree} "
          f"con R²={best_score:.4f}")

# 4) Ejecutamos la búsqueda genética para los datasets donde aplicamos la optimización
genetic_search(mat, 'Matemáticas')
genetic_search(por, 'Portugués')



# ============== Implementación de Algoritmo de Búsqueda Exhaustiva ==============


# 1) Definimos el pipeline con PolynomialFeatures
num_pipeline = Pipeline([
    ('imputer', SimpleImputer()),               
    ('scaler', StandardScaler()),          
    ('poly', PolynomialFeatures(include_bias=False)) 
])
cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_cols),
    ('cat', cat_pipeline, cat_cols)
])

pipeline = Pipeline([
    ('preproc', preprocessor),
    ('regressor', LinearRegression())
])

# 2) Definimos parámetros para el Grid Search exhaustiva
param_grid = {
    'preproc__num__imputer__strategy': ['mean', 'median'],
    'preproc__num__scaler': [StandardScaler(), MinMaxScaler()],
    'preproc__num__poly__degree': [1, 2]
}

def run_grid_search(df, name):
    X = df.drop('G3', axis=1)
    y = df['G3']
    
    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring='r2',
        n_jobs=-1,
        verbose=1
    )
    grid.fit(X, y)
    
    print(f"\n=== {name} Dataset: Grid Search Exhaustivo ===")
    print("Mejores hiperparámetros:", grid.best_params_)
    print("Mejor R² (CV):", grid.best_score_)

# 3) Ejecutamos la Búsqueda Exhaustiva para ambos Datasets
run_grid_search(mat, 'Math')
run_grid_search(por, 'Portuguese')
