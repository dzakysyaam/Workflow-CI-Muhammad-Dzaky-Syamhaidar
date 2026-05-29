import os
import json
import pandas as pd
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

df = pd.read_csv("dataset_preprocessing/dataset_preprocessed.csv")

X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

run_id_env = os.environ.get("MLFLOW_RUN_ID")

if run_id_env:
    mlflow.start_run(run_id=run_id_env)
else:
    mlflow.start_run(run_name="CI_RandomForest_Model")

try:
    mlflow.set_tag("mlflow.runName", "CI_RandomForest_Model")

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    mlflow.log_param("model", "RandomForestClassifier")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 10)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    mlflow.sklearn.log_model(model, artifact_path="model")

    cm = confusion_matrix(y_test, y_pred)

    metric_info = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True)
    }

    with open("metric_info.json", "w") as f:
        json.dump(metric_info, f, indent=4)

    mlflow.log_artifact("metric_info.json")

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title("CI Training Confusion Matrix")
    plt.savefig("training_confusion_matrix.png")
    plt.close()

    mlflow.log_artifact("training_confusion_matrix.png")

    active_run = mlflow.active_run()
    with open("run_id.txt", "w") as f:
        f.write(active_run.info.run_id)

    print("CI training selesai.")
    print("Run ID:", active_run.info.run_id)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

finally:
    mlflow.end_run()
