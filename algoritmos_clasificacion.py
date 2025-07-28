import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn_genetic import GASearchCV
from sklearn_genetic.space import Integer, Categorical, Continuous
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import SelectFromModel 
import time
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

class StudentPerformanceClassifier:
    """
    Evaluador de modelos optimizado para el dataset Student Performance.

    Selecciona 5 algoritmos de clasificación más adecuados para predecir si un estudiante aprueba o no:
    - Random Forest: Excelente para datasets con características categóricas y numéricas mixtas
    - XGBoost: Muy efectivo para problemas de clasificación con datos tabulares
    - Regresión Logística: Modelo lineal simple, rápido y muy interpretativo
    - KNN: Funciona bien cuando hay patrones locales en el rendimiento estudiantil
    - SVC: Efectivo para relaciones no lineales complejas
    """

    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        # Solo 5 modelos más relevantes para student performance
        self.models = {
            'RandomForestClassifier': RandomForestClassifier(random_state=42),
            'XGBClassifier': XGBClassifier(random_state=42, verbosity=0),
            'LogisticRegression': LogisticRegression(max_iter=1000),
            'KNeighborsClassifier': KNeighborsClassifier(),
            'SVC': SVC(probability=True)
        }

        self.param_grids_genetic = self._get_param_grids_genetic()
        self.param_grids_exhaustive = self._get_param_grids_exhaustive()

    def _get_param_grids_genetic(self):
        """
        Hiperparámetros optimizados para el dataset Student Performance.
        Consideraciones:
        - Dataset pequeño, evitar overfitting
        - Mix de variables categóricas y numéricas
        - Target binario ('Pass', 'Fail')
        """
        return {
            'RandomForestClassifier': {
                "clf__n_estimators": Integer(50, 200),
                "clf__max_depth": Integer(3, 15),
                'clf__min_samples_split': Integer(2, 20),
                'clf__min_samples_leaf': Integer(1, 10),
                'clf__max_features': Categorical(['sqrt', 'log2']),
                'clf__random_state': Categorical([42])
            },
            'XGBClassifier': {
                'clf__learning_rate': Continuous(0.01, 0.3),
                'clf__n_estimators': Integer(50, 300),
                'clf__max_depth': Integer(3, 8),
                'clf__subsample': Continuous(0.6, 1.0),
                'clf__colsample_bytree': Continuous(0.6, 1.0),
                'clf__reg_alpha': Continuous(0, 1.0),
                'clf__reg_lambda': Continuous(0, 1.0),
                'clf__random_state': Categorical([42])
            },
            'LogisticRegression': {
                'clf__C': Continuous(0.01, 10.0),
                'clf__penalty': Categorical(['l2']),
                'clf__solver': Categorical(['lbfgs', 'liblinear']),
                'clf__fit_intercept': Categorical([True])
            },
            'KNeighborsClassifier': {
                'clf__n_neighbors': Integer(3, 15),
                'clf__weights': Categorical(['uniform', 'distance']),
                'clf__algorithm': Categorical(['auto', 'ball_tree', 'kd_tree']),
                'clf__metric': Categorical(['euclidean', 'manhattan'])
            },
            'SVC': {
                'clf__C': Continuous(0.1, 100.0),
                'clf__kernel': Categorical(['rbf', 'linear', 'poly']),
                'clf__gamma': Categorical(['scale', 'auto']),
                'clf__probability': Categorical([True])
            }
        }

    def _get_param_grids_exhaustive(self):
        """Hiperparámetros para búsqueda exhaustiva - versión reducida para eficiencia."""
        return {
            'RandomForestClassifier': {
                "clf__n_estimators": [50, 100, 200],
                "clf__max_depth": [5, 10, 15],
                'clf__min_samples_split': [2, 10, 20],
                'clf__min_samples_leaf': [1, 5],
                'clf__max_features': ['sqrt', 'log2'],
                'clf__random_state': [42]
            },
            'XGBClassifier': {
                'clf__learning_rate': [0.01, 0.1, 0.2],
                'clf__n_estimators': [50, 100, 200],
                'clf__max_depth': [3, 6, 8],
                'clf__subsample': [0.8, 1.0],
                'clf__colsample_bytree': [0.8, 1.0],
                'clf__reg_alpha': [0, 0.1],
                'clf__random_state': [42]
            },
            'LogisticRegression': {
                'clf__C': [0.1, 1.0, 10.0],
                'clf__penalty': ['l2'],
                'clf__solver': ['lbfgs', 'liblinear']
            },
            'KNeighborsClassifier': {
                'clf__n_neighbors': [3, 5, 7, 10, 15],
                'clf__weights': ['uniform', 'distance'],
                'clf__algorithm': ['auto', 'ball_tree'],
                'clf__metric': ['euclidean', 'manhattan']
            },
            'SVC': {
                'clf__C': [0.1, 1.0, 10.0],
                'clf__kernel': ['rbf', 'linear'],
                'clf__gamma': ['scale', 'auto']
            }
        }
    
    

    def genetic_search(self):
        """Búsqueda genética optimizada para Student Performance."""
        results = {}

        for name, model in self.models.items():
            print(f"Entrenando {name} con método genético...")

            # Feature selection con LogisticRegressionCV
            selector = LogisticRegressionCV(cv=5, max_iter=1000, random_state=42)
            f_selection = SelectFromModel(selector)

            # Pipeline con selección de características y modelo
            pl = Pipeline([
                ('fs', f_selection),
                ('clf', model)
            ])

            # Configuración del algoritmo genético optimizada para student performance
            evolved_estimator = GASearchCV(
                estimator=pl,
                cv=5,
                scoring="f1_weighted",
                population_size=15,
                generations=8,
                tournament_size=3,
                elitism=True,
                crossover_probability=0.8,
                mutation_probability=0.15,
                param_grid=self.param_grids_genetic[name],
                algorithm="eaSimple",
                n_jobs=-1,
                verbose=True
                #random_state=42 -- No lo esta soportando
            )

            start_time = time.time()
            evolved_estimator.fit(self.X_train, self.y_train)
            end_time = time.time()

            # Evaluación en conjunto de prueba
            y_pred = evolved_estimator.predict(self.X_test)
            acc = accuracy_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred, average='weighted')

            results[name] = {
                'best_params': evolved_estimator.best_params_,
                'estimator': evolved_estimator.best_estimator_,
                'accuracy': acc,
                'f1_score': f1,
                'training_time': end_time - start_time,
                'cv_score': evolved_estimator.best_score_,
                'report': classification_report(self.y_test, y_pred)
            }

            print(f"{name} - Accuracy: {acc:.4f}, F1 Score: {f1:.4f}, Tiempo: {end_time - start_time:.2f}s")
            
            """
            # Matriz de confusión
            cm = confusion_matrix(self.y_test, y_pred)
            plt.figure(figsize=(5,4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f"Matriz de Confusión - {name}")
            plt.xlabel("Predicción")
            plt.ylabel("Real")
            plt.tight_layout()
            plt.show()

            # Curva ROC 
            if hasattr(evolved_estimator, "predict_proba"):
                y_proba = evolved_estimator.predict_proba(self.X_test)[:, 1]
                fpr, tpr, _ = roc_curve(self.y_test, y_proba)
                roc_auc = auc(fpr, tpr)

                plt.figure()
                plt.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc:.2f})")
                plt.plot([0, 1], [0, 1], 'k--')  # Línea diagonal
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title(f"Curva ROC - {name}")
                plt.legend(loc="lower right")
                plt.tight_layout()
                plt.show()
                """

        return results
    
    
    def exhaustive_search(self):
        """Búsqueda exhaustiva (GridSearchCV) optimizada para Student Performance - Clasificación."""
        results = {}

        for name, model in self.models.items():
            print(f"Entrenando {name} con método exhaustivo...")

            # Feature selection con LogisticRegressionCV
            selector = LogisticRegressionCV(cv=5, max_iter=1000, random_state=42)
            f_selection = SelectFromModel(selector)

            # Pipeline con selección de características y modelo
            pl = Pipeline([
                ('fs', f_selection),
                ('clf', model)
            ])

            # GridSearchCV con hiperparámetros definidos para cada modelo
            grid_search = GridSearchCV(
                estimator=pl,
                param_grid=self.param_grids_exhaustive[name],
                cv=5,
                scoring='f1_weighted',
                n_jobs=-1,
                verbose=1
            )

            start_time = time.time()
            grid_search.fit(self.X_train, self.y_train)
            end_time = time.time()

            # Evaluación en conjunto de prueba
            y_pred = grid_search.predict(self.X_test)
            acc = accuracy_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred, average='weighted')

            results[name] = {
                'best_params': grid_search.best_params_,
                'estimator': grid_search.best_estimator_,
                'accuracy': acc,
                'f1_score': f1,
                'training_time': end_time - start_time,
                'cv_score': grid_search.best_score_,
                'report': classification_report(self.y_test, y_pred)
            }

            print(f"{name} - Accuracy: {acc:.4f}, F1 Score: {f1:.4f}, Tiempo: {end_time - start_time:.2f}s")

        return results

    def compare_results(self, genetic_results, exhaustive_results):
        """Compara los resultados de ambos métodos de búsqueda para clasificación."""
        comparison = []

        for model_name in self.models.keys():
            genetic = genetic_results.get(model_name, {})
            exhaustive = exhaustive_results.get(model_name, {})

            # Elegir el mejor método según F1-score o tiempo si hay empate
            f1_genetic = genetic.get('f1_score', 0)
            f1_exhaustive = exhaustive.get('f1_score', 0)
            time_genetic = genetic.get('training_time', float('inf'))
            time_exhaustive = exhaustive.get('training_time', float('inf'))

            if f1_genetic > f1_exhaustive:
                best_method = 'Genetic'
            elif f1_exhaustive > f1_genetic:
                best_method = 'Exhaustive'
            else:
                # En caso de empate, elegimos el más rápido
                best_method = 'Genetic' if time_genetic < time_exhaustive else 'Exhaustive'

            comparison.append({
                'Model': model_name,
                'Genetic_Accuracy': genetic.get('accuracy', None),
                'Exhaustive_Accuracy': exhaustive.get('accuracy', None),
                'Genetic_F1': f1_genetic,
                'Exhaustive_F1': f1_exhaustive,
                'Genetic_Time': time_genetic,
                'Exhaustive_Time': time_exhaustive,
                'Best_Method': best_method
            })

        return pd.DataFrame(comparison)



if __name__ == "__main__":
    # Cargar dataset
    df = pd.read_csv("data/student_performance_dataset.csv")

    # Eliminar columnas irrelevantes
    df = df.drop(columns=["Student_ID"])

    # Codificar variables categóricas
    from sklearn.preprocessing import LabelEncoder
    categorical_cols = df.select_dtypes(include="object").columns.drop("Pass_Fail")
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    # Codificar target
    target_encoder = LabelEncoder()
    df["Pass_Fail"] = target_encoder.fit_transform(df["Pass_Fail"])

    # Separar datos
    X = df.drop(columns=["Pass_Fail"])
    y = df["Pass_Fail"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Crear evaluador y ejecutar
    evaluator = StudentPerformanceClassifier(X_train, X_test, y_train, y_test)
    genetic_results = evaluator.genetic_search()

    # Si deseas agregar exhaustive_search en el futuro, puedes agregarlo aquí
    exhaustive_results = evaluator.exhaustive_search()
    comparison = evaluator.compare_results(genetic_results, exhaustive_results)
    print(comparison)
