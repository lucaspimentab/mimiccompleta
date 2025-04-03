import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import shap
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score,
                             recall_score, f1_score, classification_report, confusion_matrix)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

warnings.filterwarnings("ignore")

df = pd.read_csv("dataset_modelagem_completo.csv")
df["gender"] = df["gender"].map({"F": 0, "M": 1})

df = df.loc[:, df.isnull().mean() < 0.7]


X = df.drop(columns=["subject_id", "target", "stay_id"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

pipeline = ImbPipeline(steps=[
    ("imputer", IterativeImputer(estimator=LinearRegression(), max_iter=10, random_state=42)),
    ("scaler", StandardScaler()),
    ("smote", SMOTE(random_state=42)),
    ("model", XGBClassifier(eval_metric='logloss', random_state=42))
])

# Hiperparâmetros
param_grid = {
    "model__n_estimators": [100, 300],
    "model__max_depth": [3, 5],
    "model__learning_rate": [0.01, 0.03, 0.1],
    "model__subsample": [0.8, 0.9],
    "model__colsample_bytree": [0.8, 0.9]
}


X_train_grid, X_val_grid, y_train_grid, y_val_grid = train_test_split(
    X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
)

# GridSearchCV
grid_search = GridSearchCV(pipeline, param_grid, scoring='roc_auc', cv=3, n_jobs=-1, verbose=1)
grid_search.fit(X_train_grid, y_train_grid)


best_model = grid_search.best_estimator_
y_prob = best_model.predict_proba(X_test)[:, 1]

best_threshold = 0.36
y_pred_best = (y_prob > best_threshold).astype(int)

# Avaliação
roc_auc = roc_auc_score(y_test, y_prob)
accuracy = accuracy_score(y_test, y_pred_best)
precision = precision_score(y_test, y_pred_best)
recall = recall_score(y_test, y_pred_best)
f1 = f1_score(y_test, y_pred_best)

print(f"AUC: {roc_auc:.4f}")
print(f"Acurácia: {accuracy:.4f}")
print(f"Precisão: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred_best, digits=3))

# Matriz de Confusão
cm = confusion_matrix(y_test, y_pred_best)
print("\n📊 Matriz de Confusão:")
print(cm)

fig, ax = plt.subplots(figsize=(5, 4))
cax = ax.matshow(cm, cmap='Blues')
for i in range(2):
    for j in range(2):
        ax.text(j, i, f'{cm[i, j]}', ha='center', va='center', color='black')
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(['Sobrevivente', 'Óbito'])
ax.set_yticklabels(['Sobrevivente', 'Óbito'])
plt.xlabel('Predito')
plt.ylabel('Real')
plt.title('Matriz de Confusão')
fig.colorbar(cax)
plt.tight_layout()
plt.savefig("matriz_confusao.png")
plt.close()

# 13. SHAP - Exibir variáveis e suas importâncias, ignorando as irrelevantes
explainer = shap.Explainer(best_model.named_steps['model'], X_train)
shap_values = explainer(X_test)

# Organizar as importâncias e filtrar as variáveis com importância maior que zero
shap_importances = pd.DataFrame({
    'Feature': X_test.columns,
    'Importance': np.abs(shap_values.values).mean(axis=0)
})

# Verificar a importância específica de 'idade'
idade_importance = shap_importances[shap_importances['Feature'] == 'idade']
print(f"\n📊 Importância da variável 'idade':")
print(idade_importance)

# Filtrar variáveis com importância maior que zero
shap_importances_filtered = shap_importances[shap_importances['Importance'] > 0]

# Ordenar as variáveis pela importância
shap_importances_filtered = shap_importances_filtered.sort_values(by='Importance', ascending=False)

# Exibir no console as variáveis com importância maior que zero
print("\n📊 Importância das variáveis (somente as relevantes):")
print(shap_importances_filtered)

# Plotar o gráfico de barras com as variáveis mais importantes
shap.summary_plot(shap_values, X_test, plot_type="bar", max_display=20)


# Falsos positivos e negativos
df_preds = pd.DataFrame(X_test)
df_preds['y_real'] = y_test.values
df_preds['y_prob'] = y_prob
df_preds['y_pred'] = y_pred_best
df_preds['erro'] = df_preds['y_real'] != df_preds['y_pred']

falsos_negativos = df_preds[(df_preds['y_real'] == 1) & (df_preds['y_pred'] == 0)]
falsos_positivos = df_preds[(df_preds['y_real'] == 0) & (df_preds['y_pred'] == 1)]
falsos_negativos.to_csv("falsos_negativos.csv", index=False)
falsos_positivos.to_csv("falsos_positivos.csv", index=False)