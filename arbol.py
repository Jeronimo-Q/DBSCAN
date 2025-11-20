import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

def entrenar_arbol(data):

    le_lsoa = LabelEncoder()
    le_crime = LabelEncoder()

    # Crear Month_num
    data["Month_num"] = data["Month"].str[-2:].astype(int)

    # Codificar variables categóricas
    data["LSOA_code_encoded"] = le_lsoa.fit_transform(data["LSOA code"])
    data["Crime_type_encoded"] = le_crime.fit_transform(data["Crime type"])

    # Eliminar ruido (-1)
    data_filtrada = data[data['Cluster'] != -1]

    # Variables de entrada
    X = data_filtrada[['Latitude', 'Longitude', 'Month_num',
                       'LSOA_code_encoded', 'Crime_type_encoded']]

    # Variable objetivo
    y = data_filtrada['Cluster']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Modelo
    modelo = DecisionTreeClassifier(
        criterion='entropy',
        max_depth=5,
        random_state=42
    )

    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    # --- MÉTRICAS ---
    print("Reporte de Clasificación:\n")
    print(classification_report(y_test, y_pred))
    print("Precisión del modelo:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")

    print("Matriz de Confusión:")
    print(confusion_matrix(y_test, y_pred))

    # ----- Gráfico con métricas -----
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    macro_f1 = report["macro avg"]["f1-score"]
    macro_precision = report["macro avg"]["precision"]
    macro_recall = report["macro avg"]["recall"]

    weighted_f1 = report["weighted avg"]["f1-score"]
    weighted_precision = report["weighted avg"]["precision"]
    weighted_recall = report["weighted avg"]["recall"]

    plt.figure(figsize=(7, 4))
    textstr = (
        f"Accuracy: {accuracy:.4f}\n"
        f"Macro Avg → Prec: {macro_precision:.2f} | Rec: {macro_recall:.2f} | F1: {macro_f1:.2f}\n"
        f"Weighted Avg → Prec: {weighted_precision:.2f} | Rec: {weighted_recall:.2f} | F1: {weighted_f1:.2f}"
    )

    plt.text(0.01, 0.5, textstr, fontsize=12, verticalalignment='center',
             bbox=dict(facecolor='lightgrey', alpha=0.3, boxstyle='round,pad=1'))
    plt.axis('off')
    plt.title("Métricas del Modelo de Árbol de Decisión")
    plt.tight_layout()
    plt.show()

    # ----- Importancia de variables -----
    feature_importances = pd.DataFrame({
        "Variable": X.columns,
        "Importancia": modelo.feature_importances_
    }).sort_values(by="Importancia", ascending=False)

    print("\nImportancia de las variables:")
    print(feature_importances)

    plt.figure(figsize=(10, 6))
    plt.barh(feature_importances["Variable"], feature_importances["Importancia"])
    plt.xlabel("Importancia")
    plt.ylabel("Variable")
    plt.title("Importancia de Variables en el Árbol de Decisión")
    plt.gca().invert_yaxis()
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

    # Devolver modelo ya entrenado
    return modelo
