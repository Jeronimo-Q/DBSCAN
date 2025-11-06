from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn import tree
import matplotlib.pyplot as plt
import pandas as pd

def entrenar_arbol_decision(df):
    """
    Entrena y evalúa un Árbol de Decisión para predecir el tipo de crimen.
    Se asume que df contiene las columnas:
    ['Month', 'Longitude', 'Latitude', 'LSOA code', 'Crime type'].
    """

    df["Month_num"] = df["Month"].str[-2:].astype(int)

    le_lsoa = LabelEncoder()
    le_crime = LabelEncoder()

    df["LSOA_code_encoded"] = le_lsoa.fit_transform(df["LSOA code"])
    df["Crime_type_encoded"] = le_crime.fit_transform(df["Crime type"])

    X = df[["Longitude", "Latitude", "Month_num", "LSOA_code_encoded"]]
    y = df["Crime_type_encoded"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    model = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=8,
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("Precisión (accuracy):", round(acc, 4))
    print("\nMatriz de confusión:\n", confusion_matrix(y_test, y_pred))
    print("\nReporte de clasificación:\n", classification_report(y_test, y_pred, target_names=le_crime.classes_))

    feature_importances = pd.DataFrame({
        "Variable": X.columns,
        "Importancia": model.feature_importances_
    }).sort_values(by="Importancia", ascending=False)

    print("\nImportancia de las variables:")
    print(feature_importances)

    from sklearn import tree
    plt.figure(figsize=(16,8))
    tree.plot_tree(
        model,
        feature_names=X.columns,
        class_names=le_crime.classes_,
        filled=True,
        rounded=True,
        fontsize=9,
        max_depth=3  
    )
    plt.show()


    return model, feature_importances
