import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_validate, RandomizedSearchCV
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

def main():
    print("Iniciando treinamento dos modelos")
    
    train_path = os.path.join("data", "processed", "train_processed.csv")
    test_path = os.path.join("data", "processed", "test_processed.csv")

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]

    X_test = df_test.drop(columns=["target"])
    y_test = df_test["target"]

    # 5 folds para validacao cruzada
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring_metrics = ["f1", "recall", "precision"]

    # 1. Dummy Classifier (Baseline)
    print("\nTreinando Dummy Classifier (Baseline)")
    dummy_model = DummyClassifier(strategy="most_frequent")
    cv_dummy = cross_validate(dummy_model, X_train, y_train, cv=cv, scoring=scoring_metrics)
    print(f"F1 CV: {cv_dummy['test_f1'].mean():.4f}")
    print(f"Recall CV: {cv_dummy['test_recall'].mean():.4f}")

    # 2. Regressao Logistica
    print("\nTreinando Regressao Logistica")
    lr_pipeline = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42))
    ])
    cv_lr = cross_validate(lr_pipeline, X_train, y_train, cv=cv, scoring=scoring_metrics)
    print(f"F1 CV: {cv_lr['test_f1'].mean():.4f}")
    print(f"Recall CV: {cv_lr['test_recall'].mean():.4f}")

    # 3. Random Forest
    print("\nTreinando Random Forest")
    rf_model = RandomForestClassifier(random_state=42, n_jobs=-1)
    cv_rf = cross_validate(rf_model, X_train, y_train, cv=cv, scoring=scoring_metrics)
    print(f"F1 CV: {cv_rf['test_f1'].mean():.4f}")
    print(f"Recall CV: {cv_rf['test_recall'].mean():.4f}")

    # 4. Ajuste de Hiperparametros
    print("\nAjustando hiperparametros da Random Forest")
    param_dist = {
        "n_estimators": [50, 100, 150],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5, 10]
    }
    rf_search = RandomizedSearchCV(
        estimator=RandomForestClassifier(random_state=42, n_jobs=-1),
        param_distributions=param_dist,
        n_iter=5,
        cv=cv,
        scoring="f1",
        random_state=42,
        n_jobs=-1
    )
    rf_search.fit(X_train, y_train)
    print("Melhores parametros:", rf_search.best_params_)
    print(f"Melhor F1 CV: {rf_search.best_score_:.4f}")

    # 5. Avaliacao na Base de Teste
    print("\nAvaliando modelo final na base de teste")
    best_model = rf_search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]

    print("\nRelatorio de Classificacao:")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Ataque"]))

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, vp = cm.ravel()
    fpr = fp / (fp + tn)
    print(f"Taxa de Falsos Positivos (FPR): {fpr*100:.2f}%")

    # Salva grafico de resultados
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Normal", "Ataque"], yticklabels=["Normal", "Ataque"], ax=axes[0])
    axes[0].set_title("Matriz de Confusao")
    axes[0].set_ylabel("Valor Real")
    axes[0].set_xlabel("Valor Predito")

    fpr_roc, tpr_roc, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr_roc, tpr_roc)
    axes[1].plot(fpr_roc, tpr_roc, color="darkorange", lw=2, label=f"Curva ROC (AUC = {roc_auc:.4f})")
    axes[1].plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel("Taxa de Falsos Positivos")
    axes[1].set_ylabel("Taxa de Verdadeiros Positivos")
    axes[1].set_title("Curva ROC (Teste)")
    axes[1].legend(loc="lower right")

    plt.tight_layout()
    img_dir = os.path.join("notebooks", "m3_modelagem", "imagens")
    os.makedirs(img_dir, exist_ok=True)
    img_path = os.path.join(img_dir, "resultado_modelo_final.png")
    plt.savefig(img_path, dpi=150, bbox_inches="tight")
    print(f"Grafico salvo em: {img_path}")

    # Salva o arquivo pkl do modelo
    import joblib
    model_dir = "app"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "modelo_final.pkl")
    joblib.dump(best_model, model_path)
    print(f"Modelo salvo em: {model_path}")

if __name__ == "__main__":
    main()
