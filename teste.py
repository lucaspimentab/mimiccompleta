import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import shap
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score,
                             recall_score, f1_score, classification_report, confusion_matrix)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# Carregar os dados
df = pd.read_csv("dataset_modelagem_completo.csv")
df["gender"] = df["gender"].map({"F": 0, "M": 1})

# Remover colunas com mais de 30% de valores ausentes
df = df.loc[:, df.isnull().mean() < 0.7]

# Definir variáveis independentes e dependentes
X = df.drop(columns=["subject_id", "target", "gender"])
y = df["target"]

# Dividir em conjunto de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Preprocessamento
imputer = SimpleImputer(strategy='mean')
scaler = StandardScaler()
model = XGBClassifier(eval_metric='logloss', random_state=42, n_estimators=100, max_depth=3, learning_rate=0.1)

# Preprocessamento e treinamento do modelo
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

# Treinamento do modelo
model.fit(X_train_scaled, y_train)

# Previsões
y_prob = model.predict_proba(X_test_scaled)[:, 1]
y_pred_best = (y_prob > 0.36).astype(int)  # Ajustar o threshold conforme necessário

# Avaliação do modelo
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

# 13. SHAP - Exibir variáveis e suas importâncias
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)

# Organizar as importâncias e filtrar as variáveis com importância maior que zero
shap_importances = pd.DataFrame({
    'Feature': X_test.columns,
    'Importance': np.abs(shap_values.values).mean(axis=0)
})

# Filtrar variáveis com importância maior que zero
shap_importances_filtered = shap_importances[shap_importances['Importance'] > 0]

# Ordenar as variáveis pela importância
shap_importances_filtered = shap_importances_filtered.sort_values(by='Importance', ascending=False)

# Exibir no console as variáveis com importância maior que zero
print("\n📊 Importância das variáveis (somente as relevantes):")
print(shap_importances_filtered)

# Plotar o gráfico de barras com as variáveis mais importantes
shap.summary_plot(shap_values, X_test, plot_type="bar", max_display=20)