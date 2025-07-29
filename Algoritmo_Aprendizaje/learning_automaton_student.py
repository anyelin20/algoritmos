import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv("student-combined.csv")
for col in df.select_dtypes(include='object'):
    df[col] = LabelEncoder().fit_transform(df[col])
X = df.drop("G3", axis=1)
y = pd.cut(df["G3"], bins=[-1, 9, 13, 20], labels=[0, 1, 2])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

class LearningAutomaton:
    def __init__(self, actions, alpha=0.1):
        self.actions = actions  # nombres de los clasificadores
        self.num_actions = len(actions)
        self.probs = np.ones(self.num_actions) / self.num_actions
        self.alpha = alpha

    def select_action(self):
        return np.random.choice(self.num_actions, p=self.probs)

    def update(self, action_index, reward):
        for i in range(self.num_actions):
            if i == action_index:
                self.probs[i] += self.alpha * (reward * (1 - self.probs[i]))
            else:
                self.probs[i] -= self.alpha * (reward * self.probs[i])
        self.probs = np.clip(self.probs, 0, 1)
        self.probs /= self.probs.sum()  # normalizar

models = {
    "LogReg": LogisticRegression(max_iter=1000),
    "RandForest": RandomForestClassifier(n_estimators=100),
    "SVM": SVC(kernel='linear', probability=True),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

action_names = list(models.keys())
automaton = LearningAutomaton(action_names, alpha=0.1)


n_iterations = 50
scores = []

for i in range(n_iterations):
    action = automaton.select_action()
    model_name = action_names[action]
    model = models[model_name]

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    reward = acc  # reward es la accuracy
    automaton.update(action, reward)
    scores.append((model_name, acc))

    print(f"Iteración {i+1}: {model_name} - Acc = {acc:.3f}")


print("\nDistribución final de probabilidades (acciones):")
for name, prob in zip(action_names, automaton.probs):
    print(f"{name}: {prob:.3f}")

mejor_modelo = action_names[np.argmax(automaton.probs)]
print(f"\n Modelo más favorecido por el autómata: {mejor_modelo}")
