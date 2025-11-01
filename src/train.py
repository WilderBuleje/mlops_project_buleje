import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import mlflow
import mlflow.sklearn
import os
import joblib

def train_models():
    # Cargar los datasets procesados
    train_path = 'data/processed/train_clean.csv'
    test_path = 'data/processed/test_clean.csv'

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError("No se encontraron los archivos train_clean.csv o test_clean.csv. Ejecuta primero preprocessing.py")

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    # Definir variables independientes y dependientes
    X_train = df_train.iloc[:, 3::]
    y_train = df_train.iloc[:, 2]

    X_test = df_test.iloc[:, 3::]
    y_test = df_test.iloc[:, 2]

    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    # Configurar experimento MLflow
    mlflow.set_experiment("mlops_demo")

    # ======== MODELO 1: Regresión Logística ========
    with mlflow.start_run(run_name="LogisticRegression"):
        log_model = LogisticRegression(max_iter=1000)
        log_model.fit(X_train, y_train)

        # Guardar modelo local
        os.makedirs("models", exist_ok=True)
        joblib.dump(log_model, "models/logistic_regression.pkl")
        
        # Predicciones
        y_pred_log = log_model.predict(X_test)

        # Métricas
        acc_log = accuracy_score(y_test, y_pred_log)
        f1_log = f1_score(y_test, y_pred_log, average='weighted')

        # Registrar en MLflow
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_metric("accuracy", acc_log)
        mlflow.log_metric("f1_score", f1_log)
        mlflow.sklearn.log_model(log_model, "model")

        print(f"Logistic Regression - Accuracy: {acc_log:.4f}, F1: {f1_log:.4f}")

    # ======== MODELO 2: Árbol de Decisión ========
    with mlflow.start_run(run_name="DecisionTree"):
        tree_model = DecisionTreeClassifier(random_state=42)
        tree_model.fit(X_train, y_train)

        # Guardar modelo local
        os.makedirs("models", exist_ok=True)
        joblib.dump(tree_model, "models/decision_tree.pkl")

        # Predicciones
        y_pred_tree = tree_model.predict(X_test)

        # Métricas
        acc_tree = accuracy_score(y_test, y_pred_tree)
        f1_tree = f1_score(y_test, y_pred_tree, average='weighted')

        # Registrar en MLflow
        mlflow.log_param("model_type", "DecisionTree")
        mlflow.log_metric("accuracy", acc_tree)
        mlflow.log_metric("f1_score", f1_tree)
        mlflow.sklearn.log_model(tree_model, "model")

        print(f"Decision Tree - Accuracy: {acc_tree:.4f}, F1: {f1_tree:.4f}")

if __name__ == "__main__":
    train_models()