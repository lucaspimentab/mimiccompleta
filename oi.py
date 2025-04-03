import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, cross_val_predict
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# 1. Carregar e preparar os dados
df = pd.read_csv("Dataset.csv")
df["gender"] = df["gender"].map({"F": 0, "M": 1})
df["charlson_grave"] = (df["charlson"] > 7).astype(int)
df["sapsii_log"] = np.log1p(df["sapsii"])
df["interacao_oasis_sofa"] = df["oasis"] * df["sofa_24hours"]
df["idade_ao_quadrado"] = df["idade"] ** 2

X = df.drop(columns=["subject_id", "target", "gender", "stay_id"])
y = df["target"]

# 2. Imputar e escalar os dados inteiros
imputer = SimpleImputer(strategy="mean")
X_imp = imputer.fit_transform(X)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imp)

# 3. Definir modelo com melhores hiperparâmetros
model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_estimators=300,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.9,
    colsample_bytree=0.9,
    gamma=1,
    min_child_weight=1,
    reg_lambda=1,
    reg_alpha=0
)

# 4. Avaliação com validação cruzada
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
y_prob_cv = cross_val_predict(model, X_scaled, y, cv=cv, method='predict_proba')[:, 1]
threshold = 0.33
y_pred_cv = (y_prob_cv > threshold).astype(int)

# 5. Métricas globais (agregadas)
roc_auc = roc_auc_score(y, y_prob_cv)
precision = precision_score(y, y_pred_cv)
recall = recall_score(y, y_pred_cv)
f1 = f1_score(y, y_pred_cv)
accuracy = accuracy_score(y, y_pred_cv)

print(f"🧠 AUC ROC (cross-val): {roc_auc:.4f}")
print(f"📉 Matriz de Confusão:\n{confusion_matrix(y, y_pred_cv)}")
print(f"\n🎯 Acurácia: {accuracy:.2f}")
print(f"📊 Precisão: {precision:.2f}")
print(f"📊 Recall: {recall:.2f}")
print(f"📊 F1-Score: {f1:.2f}")

# 6. Classification Report (global, sem separar as classes)
report = classification_report(y, y_pred_cv, output_dict=True)
report_df = pd.DataFrame(report).transpose().round(2)

# Exibir o Classification Report global (não precisa de remoção de colunas)
print("\n📊 Classification Report (Global):")
print(report_df.to_string())

# 7. Matriz de confusão (imagem)
cm = confusion_matrix(y, y_pred_cv)
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
plt.title('Matriz de Confusão (CV)')
fig.colorbar(cax)
plt.tight_layout()
plt.savefig("matriz_confusao_cv.png")
plt.close()

# 8. SHAP - Importância das variáveis ordenadas
model.fit(X_scaled, y)
explainer = shap.Explainer(model)
shap_values = explainer(X_scaled)
shap_abs = np.abs(shap_values.values).mean(axis=0)
features = X.columns
shap_df = pd.DataFrame({
    'feature': features,
    'mean_abs_shap': shap_abs
}).sort_values(by='mean_abs_shap', ascending=True)

plt.figure(figsize=(8, 10))
plt.barh(shap_df['feature'], shap_df['mean_abs_shap'])
plt.xlabel('Importância média (|SHAP|)')
plt.title('Importância das variáveis - SHAP (CV)')
plt.tight_layout()
plt.savefig("shap_barplot_cv.png")
plt.close()

# 9. Exportar falsos positivos e negativos
df_preds = pd.DataFrame(X)
df_preds['y_real'] = y.values
df_preds['y_prob'] = y_prob_cv
df_preds['y_pred'] = y_pred_cv
df_preds['erro'] = df_preds['y_real'] != df_preds['y_pred']

falsos_negativos = df_preds[(df_preds['y_real'] == 1) & (df_preds['y_pred'] == 0)]
falsos_positivos = df_preds[(df_preds['y_real'] == 0) & (df_preds['y_pred'] == 1)]

falsos_negativos.to_csv("falsos_negativos_cv.csv", index=False)
falsos_positivos.to_csv("falsos_positivos_cv.csv", index=False)