import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn_genetic import GASearchCV
from sklearn_genetic.space import Integer, Categorical, Continuous
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.feature_selection import SelectFromModel 
import time
from sklearn.linear_model import LassoCV 
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
import os
from datetime import datetime


class StudentPerformanceEvaluator:
    """
    Evaluador de modelos optimizado para el dataset Student Performance de UCI.
    
    Selecciona 5 algoritmos de regresión más adecuados para predecir el rendimiento estudiantil:
    - Random Forest: Excelente para datasets con características categóricas y numéricas mixtas
    - XGBoost: Muy efectivo para problemas de regresión con datos tabulares
    - Ridge: Regularización L2, buena para evitar overfitting con múltiples características
    - KNN: Funciona bien cuando hay patrones locales en el rendimiento estudiantil  
    - SVR: Efectivo para relaciones no lineales complejas
    """

    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        # Solo 5 modelos más relevantes para student performance
        self.models = {
            'RandomForestRegressor': RandomForestRegressor(random_state=42),
            'XGBRegressor': XGBRegressor(random_state=42, verbosity=0),
            'Ridge': Ridge(),
            'KNeighborsRegressor': KNeighborsRegressor(),
            'SVR': SVR()
        }
        
        self.param_grids_genetic = self._get_param_grids_genetic()
        self.param_grids_exhaustive = self._get_param_grids_exhaustive()

    def _get_param_grids_genetic(self):
        """
        Hiperparámetros optimizados para el dataset Student Performance.
        Consideraciones:
        - Dataset pequeño (~395 estudiantes), evitar overfitting
        - Mix de variables categóricas y numéricas
        - Target variable continua (G3: 0-20)
        """
        return {
            'RandomForestRegressor': {
                "clf__n_estimators": Integer(50, 200),  # Aumentado para mejor generalización
                "clf__max_depth": Integer(3, 15),       # Rango amplio para capturar complejidad
                'clf__min_samples_split': Integer(2, 20), # Evitar overfitting en dataset pequeño
                'clf__min_samples_leaf': Integer(1, 10),   # Hojas más grandes para generalización
                'clf__max_features': Categorical(['sqrt', 'log2', 0.8]), # Reduce overfitting
                'clf__random_state': Categorical([42])
            },
            'XGBRegressor': {
                'clf__learning_rate': Continuous(0.01, 0.3),    # Rango más amplio
                'clf__n_estimators': Integer(50, 300),          # Más estimadores posibles
                'clf__max_depth': Integer(3, 8),                # Profundidad moderada
                'clf__subsample': Continuous(0.6, 1.0),         # Subsampleo para regularización
                'clf__colsample_bytree': Continuous(0.6, 1.0),  # Feature sampling
                'clf__reg_alpha': Continuous(0, 1.0),           # L1 regularization
                'clf__reg_lambda': Continuous(0, 1.0),          # L2 regularization
                'clf__random_state': Categorical([42])
            },
            'Ridge': {
                'clf__alpha': Continuous(0.1, 100.0),           # Rango amplio de regularización
                'clf__fit_intercept': Categorical([True]),       # Siempre incluir intercept
                'clf__solver': Categorical(['auto', 'svd', 'cholesky', 'lsqr']) # Más opciones de solver
            },
            'KNeighborsRegressor': {
                'clf__n_neighbors': Integer(3, 15),             # Más vecinos para dataset pequeño
                'clf__weights': Categorical(['uniform', 'distance']),
                'clf__algorithm': Categorical(['auto', 'ball_tree', 'kd_tree']),
                'clf__metric': Categorical(['euclidean', 'manhattan', 'minkowski'])
            },
            'SVR': {
                'clf__C': Continuous(0.1, 100.0),               # Parámetro de regularización
                'clf__kernel': Categorical(['rbf', 'linear', 'poly']), # Diferentes kernels
                'clf__gamma': Categorical(['scale', 'auto']),     # Parámetro del kernel RBF
                'clf__epsilon': Continuous(0.01, 1.0)           # Tolerancia para SVR
            }
        }
    
    def _get_param_grids_exhaustive(self):
        """Hiperparámetros para búsqueda exhaustiva - versión reducida para eficiencia."""
        return {
            'RandomForestRegressor': {
                "clf__n_estimators": [50, 100, 200],
                "clf__max_depth": [5, 10, 15],
                'clf__min_samples_split': [2, 10, 20],
                'clf__min_samples_leaf': [1, 5],
                'clf__max_features': ['sqrt', 'log2'],
                'clf__random_state': [42]
            },
            'XGBRegressor': {
                'clf__learning_rate': [0.01, 0.1, 0.2],
                'clf__n_estimators': [50, 100, 200],
                'clf__max_depth': [3, 6, 8],
                'clf__subsample': [0.8, 1.0],
                'clf__colsample_bytree': [0.8, 1.0],
                'clf__reg_alpha': [0, 0.1],
                'clf__random_state': [42]
            },
            'Ridge': {
                'clf__alpha': [0.1, 1.0, 10.0, 100.0],
                'clf__fit_intercept': [True],
                'clf__solver': ['auto', 'svd', 'cholesky']
            },
            'KNeighborsRegressor': {
                'clf__n_neighbors': [3, 5, 7, 10, 15],
                'clf__weights': ['uniform', 'distance'],
                'clf__algorithm': ['auto', 'ball_tree'],
                'clf__metric': ['euclidean', 'manhattan']
            },
            'SVR': {
                'clf__C': [0.1, 1.0, 10.0, 100.0],
                'clf__kernel': ['rbf', 'linear'],
                'clf__gamma': ['scale', 'auto'],
                'clf__epsilon': [0.1, 0.01]
            }
        }

    def genetic_search(self):
        """Búsqueda genética optimizada para Student Performance."""
        results = {}

        print("Realizando selección de características con LassoCV...")
        lasso_cv = LassoCV(cv=5, random_state=42)
        lasso_cv.fit(self.X_train, self.y_train)
        selector = SelectFromModel(lasso_cv, prefit=True)
        X_train_sel = selector.transform(self.X_train)
        X_test_sel = selector.transform(self.X_test)
        
        for name, model in self.models.items():
            print(f"Entrenando {name} con método genético...")
            
            
            # Pipeline con selección de características y modelo
            pl = Pipeline([
                ('fs', StandardScaler()),
                ('clf', model)
            ])
            
            # Configuración del algoritmo genético optimizada para student performance
            evolved_estimator = GASearchCV(
                estimator=pl,
                cv=5,  # 5-fold CV adecuado para dataset pequeño
                scoring="neg_mean_squared_error",
                population_size=15,  # Población más grande para mejor exploración
                generations=8,       # Más generaciones para convergencia
                tournament_size=3,
                elitism=True,
                crossover_probability=0.8,
                mutation_probability=0.15,  # Mutación ligeramente mayor
                param_grid=self.param_grids_genetic[name],
                algorithm="eaSimple",
                n_jobs=-1,
                verbose=True,
                #random_state=42  # Para reproducibilidad
            )
            
            start_time = time.time()
            evolved_estimator.fit(self.X_train, self.y_train)
            end_time = time.time()
            
            # Evaluación en conjunto de prueba
            y_pred = evolved_estimator.predict(self.X_test)
            r2 = r2_score(self.y_test, y_pred)
            mse = mean_squared_error(self.y_test, y_pred)
            rmse = np.sqrt(mse)
            
            results[name] = {
                'best_params': evolved_estimator.best_params_,
                'estimator': evolved_estimator.best_estimator_,
                'r2_score': r2,
                'mse': mse,
                'rmse': rmse,
                'training_time': end_time - start_time,
                'cv_score': evolved_estimator.best_score_
            }
            
            print(f"{name} - R²: {r2:.4f}, RMSE: {rmse:.4f}, Tiempo: {end_time - start_time:.2f}s")
        
        guardar_resultados_csv("resultados_genetico.csv", results)
        return results

    def exhaustive_search(self):
        """Búsqueda exhaustiva optimizada para Student Performance."""
        results = {}
        
        for name, model in self.models.items():
            print(f"Entrenando {name} con método exhaustivo...")
            
            # Feature selection con LassoCV
            lasso_cv = LassoCV(cv=5, random_state=42)
            lasso_cv.fit(self.X_train, self.y_train)
            f_selection = SelectFromModel(lasso_cv, prefit=True)
            
            # Pipeline simple para búsqueda exhaustiva
            pl = Pipeline([
                ('fs', SelectFromModel(LassoCV(cv=5))),
                ('clf', model)
            ])
            
            grid_search = GridSearchCV(
                estimator=pl,
                param_grid=self.param_grids_exhaustive[name],
                cv=5,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                verbose=1
            )
            
            start_time = time.time()
            grid_search.fit(self.X_train, self.y_train)
            end_time = time.time()
            
            # Evaluación en conjunto de prueba
            y_pred = grid_search.predict(self.X_test)
            r2 = r2_score(self.y_test, y_pred)
            mse = mean_squared_error(self.y_test, y_pred)
            rmse = np.sqrt(mse)
            
            results[name] = {
                'best_params': grid_search.best_params_,
                'estimator': grid_search.best_estimator_,
                'r2_score': r2,
                'mse': mse,
                'rmse': rmse,
                'training_time': end_time - start_time,
                'cv_score': grid_search.best_score_
            }
            
            print(f"{name} - R²: {r2:.4f}, RMSE: {rmse:.4f}, Tiempo: {end_time - start_time:.2f}s")

        guardar_resultados_csv("resultados_exhaustivo.csv", results)
        return results

    def compare_results(self, genetic_results, exhaustive_results):
        """Compara los resultados de ambos métodos de búsqueda."""
        comparison = []
        
        for model_name in self.models.keys():
            genetic = genetic_results[model_name]
            exhaustive = exhaustive_results[model_name]
            
            comparison.append({
                'Model': model_name,
                'Genetic_R2': genetic['r2_score'],
                'Exhaustive_R2': exhaustive['r2_score'],
                'Genetic_RMSE': genetic['rmse'],
                'Exhaustive_RMSE': exhaustive['rmse'],
                'Genetic_Time': genetic['training_time'],
                'Exhaustive_Time': exhaustive['training_time'],
                'Best_Method': 'Genetic' if genetic['r2_score'] > exhaustive['r2_score'] else 'Exhaustive'
            })
        
        return pd.DataFrame(comparison)
    


def guardar_resultados_csv(nombre_archivo, resultados_dict):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    registros = []

    for modelo, datos in resultados_dict.items():
        registros.append({
            "timestamp": now,
            "modelo": modelo,
            "r2_score": datos['r2_score'],
            "rmse": datos['rmse'],
            "cv_score": datos['cv_score'],
            "training_time": datos['training_time'],
            "best_params": str(datos['best_params'])  # convert dict to string
        })

    df_resultados = pd.DataFrame(registros)

    if os.path.exists(nombre_archivo):
        df_resultados.to_csv(nombre_archivo, mode='a', header=False, index=False)
    else:
        df_resultados.to_csv(nombre_archivo, index=False)


# Ejemplo de uso:
"""
# Cargar y preparar el dataset Student Performance
from sklearn.preprocessing import LabelEncoder

# Asumir que ya tienes el dataset cargado como 'student_data'
# Preprocesamiento necesario para variables categóricas
categorical_features = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 
                       'reason', 'guardian', 'schoolsup', 'famsup', 'paid', 'activities', 
                       'nursery', 'higher', 'internet', 'romantic']

# Codificar variables categóricas
le = LabelEncoder()
for feature in categorical_features:
    if feature in student_data.columns:
        student_data[feature] = le.fit_transform(student_data[feature])

# Separar características y variable objetivo (G3)
X = student_data.drop(['G3'], axis=1)  # Excluir G1, G2 si quieres un reto mayor
y = student_data['G3']

# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y usar el evaluador
evaluator = StudentPerformanceEvaluator(X_train, X_test, y_train, y_test)

# Ejecutar búsquedas
genetic_results = evaluator.genetic_search()
exhaustive_results = evaluator.exhaustive_search()

# Comparar resultados
comparison_df = evaluator.compare_results(genetic_results, exhaustive_results)
print(comparison_df)
"""