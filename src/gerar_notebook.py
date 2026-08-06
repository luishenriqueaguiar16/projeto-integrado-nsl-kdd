import json
import os

def create_notebook():
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Módulo 3: Modelagem e Experimentação - Detecção de Intrusões NSL-KDD\n",
                    "\n",
                    "Treinamento e avaliação de modelos de Machine Learning para classificação de conexões de rede em normais ou ataques no dataset NSL-KDD."
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 1. Importação de Ferramentas (Bibliotecas)"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import os\n",
                    "import pandas as pd\n",
                    "import numpy as np\n",
                    "import matplotlib.pyplot as plt\n",
                    "import seaborn as sns\n",
                    "from sklearn.model_selection import StratifiedKFold, cross_validate, RandomizedSearchCV\n",
                    "from sklearn.dummy import DummyClassifier\n",
                    "from sklearn.linear_model import LogisticRegression\n",
                    "from sklearn.ensemble import RandomForestClassifier\n",
                    "from sklearn.preprocessing import StandardScaler\n",
                    "from sklearn.pipeline import Pipeline\n",
                    "from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc\n",
                    "\n",
                    "sns.set_theme(style=\"whitegrid\")\n",
                    "plt.rcParams[\"figure.figsize\"] = (12, 5)"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 2. Carregamento das Bases de Dados Pré-processadas\n",
                    "\n",
                    "Carregamento das tabelas CSV de treino e teste limpas e com variáveis categóricas convertidas em colunas numéricas."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "train_path = os.path.join(\"..\", \"..\", \"data\", \"processed\", \"train_processed.csv\")\n",
                    "test_path = os.path.join(\"..\", \"..\", \"data\", \"processed\", \"test_processed.csv\")\n",
                    "\n",
                    "df_train = pd.read_csv(train_path)\n",
                    "df_test = pd.read_csv(test_path)\n",
                    "\n",
                    "print(f\"Base de Treino carregada: {df_train.shape[0]} linhas, {df_train.shape[1]} colunas.\")\n",
                    "print(f\"Base de Teste carregada: {df_test.shape[0]} linhas, {df_test.shape[1]} colunas.\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 3. Separação de Atributos Preditivos (X) e Classe Alvo (y)\n",
                    "\n",
                    "Divisão entre variáveis de entrada (X) e a coluna resposta target (y)."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "X_train = df_train.drop(columns=[\"target\"])\n",
                    "y_train = df_train[\"target\"]\n",
                    "\n",
                    "X_test = df_test.drop(columns=[\"target\"])\n",
                    "y_test = df_test[\"target\"]"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 4. Estratégia de Validação Cruzada (Cross-Validation)\n",
                    "\n",
                    "Divisão da base de treino em 5 partes estratificadas para treinamento e validação."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 5. Experimentos com Diferentes Modelos"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### Experimento 1: Dummy Classifier (Baseline)\n",
                    "\n",
                    "Modelo inicial simples focado na classe mais frequente."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "dummy = DummyClassifier(strategy=\"most_frequent\")\n",
                    "\n",
                    "scoring_metrics = [\"f1\", \"recall\", \"precision\"]\n",
                    "cv_results_dummy = cross_validate(dummy, X_train, y_train, cv=cv, scoring=scoring_metrics)\n",
                    "\n",
                    "print(\"Dummy Classifier (Baseline)\")\n",
                    "print(f\"F1-Score Medio: {cv_results_dummy['test_f1'].mean():.4f}\")\n",
                    "print(f\"Recall Medio: {cv_results_dummy['test_recall'].mean():.4f}\")\n",
                    "print(f\"Precision Media: {cv_results_dummy['test_precision'].mean():.4f}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### Experimento 2: Regressão Logística (Modelo Linear)\n",
                    "\n",
                    "Modelo linear utilizando StandardScaler para padronizar as colunas numéricas."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "lr_pipeline = Pipeline(steps=[\n",
                    "    (\"scaler\", StandardScaler()),\n",
                    "    (\"classifier\", LogisticRegression(max_iter=1000, random_state=42))\n",
                    "])\n",
                    "\n",
                    "cv_results_lr = cross_validate(lr_pipeline, X_train, y_train, cv=cv, scoring=scoring_metrics)\n",
                    "\n",
                    "print(\"Regressao Logistica\")\n",
                    "print(f\"F1-Score Medio: {cv_results_lr['test_f1'].mean():.4f}\")\n",
                    "print(f\"Recall Medio: {cv_results_lr['test_recall'].mean():.4f}\")\n",
                    "print(f\"Precision Media: {cv_results_lr['test_precision'].mean():.4f}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### Experimento 3: Random Forest (Modelo Baseado em Árvores)\n",
                    "\n",
                    "Modelo de Floresta Aleatória treinado nos dados originais sem necessidade de padronização."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "rf = RandomForestClassifier(random_state=42, n_jobs=-1)\n",
                    "\n",
                    "cv_results_rf = cross_validate(rf, X_train, y_train, cv=cv, scoring=scoring_metrics)\n",
                    "\n",
                    "print(\"Random Forest (Sem Ajustes)\")\n",
                    "print(f\"F1-Score Medio: {cv_results_rf['test_f1'].mean():.4f}\")\n",
                    "print(f\"Recall Medio: {cv_results_rf['test_recall'].mean():.4f}\")\n",
                    "print(f\"Precision Media: {cv_results_rf['test_precision'].mean():.4f}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 6. Ajuste de Hiperparâmetros\n",
                    "\n",
                    "Busca aleatória (RandomizedSearchCV) para otimizar os parâmetros da Random Forest."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "param_dist = {\n",
                    "    \"n_estimators\": [50, 100, 150],\n",
                    "    \"max_depth\": [10, 20, None],\n",
                    "    \"min_samples_split\": [2, 5, 10]\n",
                    "}\n",
                    "\n",
                    "rf_search = RandomizedSearchCV(\n",
                    "    estimator=RandomForestClassifier(random_state=42, n_jobs=-1),\n",
                    "    param_distributions=param_dist,\n",
                    "    n_iter=5,\n",
                    "    cv=cv,\n",
                    "    scoring=\"f1\",\n",
                    "    random_state=42,\n",
                    "    n_jobs=-1\n",
                    ")\n",
                    "\n",
                    "print(\"Iniciando busca de hiperparametros...\")\n",
                    "rf_search.fit(X_train, y_train)\n",
                    "\n",
                    "print(\"Melhores Hiperparametros Encontrados:\")\n",
                    "print(rf_search.best_params_)\n",
                    "print(f\"Melhor F1-Score na Validacao: {rf_search.best_score_:.4f}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 7. Avaliação do Modelo Final no Teste\n",
                    "\n",
                    "Avaliação do modelo ajustado na base de testes."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "best_model = rf_search.best_estimator_\n",
                    "y_pred = best_model.predict(X_test)\n",
                    "y_pred_proba = best_model.predict_proba(X_test)[:, 1]\n",
                    "\n",
                    "print(\"Relatorio de Classificacao na Base de Teste:\")\n",
                    "print(classification_report(y_test, y_pred, target_names=[\"Normal\", \"Ataque\"]))\n",
                    "\n",
                    "cm = confusion_matrix(y_test, y_pred)\n",
                    "tn, fp, fn, vp = cm.ravel()\n",
                    "fpr = fp / (fp + tn)\n",
                    "print(f\"Taxa de Falsos Positivos (FPR): {fpr*100:.2f}%\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 8. Visualizações Finais (Matriz de Confusão e Curva ROC)"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
                    "\n",
                    "sns.heatmap(cm, annot=True, fmt=\"d\", cmap=\"Blues\", \n",
                    "            xticklabels=[\"Normal\", \"Ataque\"], yticklabels=[\"Normal\", \"Ataque\"], ax=axes[0])\n",
                    "axes[0].set_title(\"Matriz de Confusao\")\n",
                    "axes[0].set_ylabel(\"Valor Real\")\n",
                    "axes[0].set_xlabel(\"Valor Predito\")\n",
                    "\n",
                    "fpr_roc, tpr_roc, _ = roc_curve(y_test, y_pred_proba)\n",
                    "roc_auc = auc(fpr_roc, tpr_roc)\n",
                    "\n",
                    "axes[1].plot(fpr_roc, tpr_roc, color=\"darkorange\", lw=2, label=f\"Curva ROC (AUC = {roc_auc:.4f})\")\n",
                    "axes[1].plot([0, 1], [0, 1], color=\"navy\", lw=2, linestyle=\"--\")\n",
                    "axes[1].set_xlim([0.0, 1.0])\n",
                    "axes[1].set_ylim([0.0, 1.05])\n",
                    "axes[1].set_xlabel(\"Taxa de Falsos Positivos\")\n",
                    "axes[1].set_ylabel(\"Taxa de Verdadeiros Positivos\")\n",
                    "axes[1].set_title(\"Curva ROC\")\n",
                    "axes[1].legend(loc=\"lower right\")\n",
                    "\n",
                    "plt.tight_layout()\n",
                    "os.makedirs(\"imagens\", exist_ok=True)\n",
                    "plt.savefig(\"imagens/resultado_modelo_final.png\", dpi=150, bbox_inches=\"tight\")\n",
                    "plt.show()"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (.venv)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.13.2"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }

    os.makedirs(os.path.join("notebooks", "m3_modelagem"), exist_ok=True)
    nb_path = os.path.join("notebooks", "m3_modelagem", "modelagem_experimentacao.ipynb")

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    
    print(f"Notebook gerado em: {nb_path}")

if __name__ == "__main__":
    create_notebook()
