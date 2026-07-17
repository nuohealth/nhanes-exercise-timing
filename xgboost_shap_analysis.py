#!/usr/bin/env python3
"""
XGBoost + SHAP 分析 — 运动时机与代谢综合征
少阳·肝木·系统建模
2026-07-16

替代方案：XGBoost替代Logistic回归（主分析）+ SHAP解释
5-fold CV: AUC, Precision, Recall, F1
SHAP: summary plot (全局), dependence plot (timing最重要)
对比: XGBoost SHAP vs Logistic OR

数据源: NHANES 2011-2014 (运动时机论文最终分析数据)
"""

import pandas as pd
import numpy as np
import os
import warnings
import json
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')  # 无显示后端
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.metrics import (roc_auc_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_curve,
                             accuracy_score, classification_report)
from sklearn.preprocessing import LabelEncoder

import xgboost as xgb
import shap

# ─−− 配置 ─−−
PROJECT_ROOT = '/root/.openclaw/workspace/projects/exercise-timing-paper'
DATA_PATH    = os.path.join(PROJECT_ROOT, '03_analysis', 'final_analysis_data.csv')
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, 'review', 'round1')
FIGURE_DIR   = os.path.join(OUTPUT_DIR, 'figures')

os.makedirs(FIGURE_DIR, exist_ok=True)

RANDOM_SEED = 42
N_FOLDS     = 5

print("=" * 70)
print("XGBoost + SHAP 分析 — 运动时机与代谢综合征")
print("=" * 70)

# ═══════════════════════════════════════════════════════════════════════════
# 1. 加载并准备数据
# ═══════════════════════════════════════════════════════════════════════════
print("\n【步骤1】加载数据...")
df = pd.read_csv(DATA_PATH)
print(f"  总样本量: {len(df)}")
print(f"  MetS阳性: {df['has_mets'].sum()} ({df['has_mets'].mean()*100:.1f}%)")

# ─−− 特征工程 ─−−
# 目标变量
y = df['has_mets'].values

# 编码 timing_group (5类)
timing_order = ['Morning', 'Noon', 'Afternoon', 'Evening', 'Mixed']
le_timing = LabelEncoder()
le_timing.fit(timing_order)
timing_encoded = le_timing.transform(df['dominant_timing'].values)

# 编码种族: RIDRETH1 (1=Mexican American, 2=Other Hispanic, 3=Non-Hispanic White,
#                   4=Non-Hispanic Black, 5=Other/Multiracial)
race_encoded = df['RIDRETH1'].values.astype(int)

# BMI缺失: 简单均值填充
bmi_median = df['BMXBMI'].median()
bmi_filled = df['BMXBMI'].fillna(bmi_median).values

# 构建特征矩阵
feature_names = [
    'Age',                    # RIDAGEYR
    'Sex',                    # RIAGENDR (1=Male, 2=Female)
    'Race',                   # RIDRETH1 编码
    'BMI',                    # BMXBMI
    'MVPA_Volume',            # Total_MVPA
    'Timing_Morning',         # reference
    'Timing_Noon',
    'Timing_Afternoon',
    'Timing_Evening',
    'Timing_Mixed',
]

# 构建方式A: one-hot timing
timing_dummies = pd.get_dummies(df['dominant_timing'], prefix='Timing')
# 确保所有列按顺序存在
for t in ['Timing_Morning', 'Timing_Noon', 'Timing_Afternoon', 'Timing_Evening', 'Timing_Mixed']:
    if t not in timing_dummies.columns:
        timing_dummies[t] = 0

X = np.column_stack([
    df['RIDAGEYR'].values,
    df['RIAGENDR'].values,
    race_encoded,
    bmi_filled,
    np.log1p(df['Total_MVPA'].values),       # log-transformed MVPA
    timing_dummies[['Timing_Morning', 'Timing_Noon', 'Timing_Afternoon',
                    'Timing_Evening', 'Timing_Mixed']].values
])

# 也构建方式B: timing作为序数 + one-hot（用于SHAP解读灵活性）
# 主要用于SHAP对timing的dependence分析
X_feature_names = feature_names

print(f"\n  特征维度: {X.shape[1]}")
print(f"  特征: {feature_names}")

# 保存用于Logistic回归对比的dataframe
df_analysis = df.copy()
for t in ['Timing_Morning', 'Timing_Noon', 'Timing_Afternoon', 'Timing_Evening', 'Timing_Mixed']:
    df_analysis[t] = timing_dummies[t].values
df_analysis['age_centered'] = (df_analysis['RIDAGEYR'] - df_analysis['RIDAGEYR'].mean()) / 10
df_analysis['bmi_centered'] = (df_analysis['BMXBMI'] - df_analysis['BMXBMI'].mean()) / 5
df_analysis['mvpa_log'] = np.log1p(df_analysis['Total_MVPA'])
df_analysis['race_black'] = (df_analysis['RIDRETH1'] == 4).astype(int)
df_analysis['race_hispanic'] = ((df_analysis['RIDRETH1'] == 1) | (df_analysis['RIDRETH1'] == 2)).astype(int)

# ═══════════════════════════════════════════════════════════════════════════
# 2. 训练XGBoost + 5折交叉验证
# ═══════════════════════════════════════════════════════════════════════════
print("\n【步骤2】XGBoost 5折交叉验证...")

# 计算scale_pos_weight
neg_count = (y == 0).sum()
pos_count = (y == 1).sum()
scale_weight = neg_count / pos_count
print(f"  scale_pos_weight = {scale_weight:.2f} ({neg_count}/{pos_count})")

xgb_params = {
    'n_estimators': 200,
    'max_depth': 4,
    'learning_rate': 0.1,
    'scale_pos_weight': scale_weight,
    'random_state': RANDOM_SEED,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'eval_metric': 'auc',
    'use_label_encoder': False,
    'verbosity': 0,
}

model = xgb.XGBClassifier(**xgb_params)

# ─−− 5折交叉验证 ─−−
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

cv_scores = {
    'auc': [], 'precision': [], 'recall': [], 'f1': [], 'accuracy': []
}
all_y_true = []
all_y_pred = []
all_y_prob = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]
    
    auc = roc_auc_score(y_val, y_prob)
    prec = precision_score(y_val, y_pred)
    rec = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    acc = accuracy_score(y_val, y_pred)
    
    cv_scores['auc'].append(auc)
    cv_scores['precision'].append(prec)
    cv_scores['recall'].append(rec)
    cv_scores['f1'].append(f1)
    cv_scores['accuracy'].append(acc)
    
    all_y_true.extend(y_val)
    all_y_pred.extend(y_pred)
    all_y_prob.extend(y_prob)
    
    print(f"  Fold {fold}: AUC={auc:.4f}, Precision={prec:.4f}, "
          f"Recall={rec:.4f}, F1={f1:.4f}, Acc={acc:.4f}")

print(f"\n  ═══ 5折CV汇总 ═══")
print(f"  AUC:        {np.mean(cv_scores['auc']):.4f} ± {np.std(cv_scores['auc']):.4f}")
print(f"  Precision:  {np.mean(cv_scores['precision']):.4f} ± {np.std(cv_scores['precision']):.4f}")
print(f"  Recall:     {np.mean(cv_scores['recall']):.4f} ± {np.std(cv_scores['recall']):.4f}")
print(f"  F1:         {np.mean(cv_scores['f1']):.4f} ± {np.std(cv_scores['f1']):.4f}")
print(f"  Accuracy:   {np.mean(cv_scores['accuracy']):.4f} ± {np.std(cv_scores['accuracy']):.4f}")

all_y_true = np.array(all_y_true)
all_y_pred = np.array(all_y_pred)
all_y_prob = np.array(all_y_prob)

# ─−− 混淆矩阵（合并所有折） ─−−
cm = confusion_matrix(all_y_true, all_y_pred)
print(f"\n  混淆矩阵 (所有折合并):")
print(f"      TN={cm[0,0]}, FP={cm[0,1]}")
print(f"      FN={cm[1,0]}, TP={cm[1,1]}")

# ═══════════════════════════════════════════════════════════════════════════
# 3. 完整模型 (full-data) 用于SHAP解释
# ═══════════════════════════════════════════════════════════════════════════
print("\n【步骤3】在全量数据上训练完整模型（用于SHAP解释）...")

full_model = xgb.XGBClassifier(**xgb_params)
full_model.fit(X, y)

# ─−− 特征重要性（原生XGBoost） ─−−
importance_dict = full_model.get_booster().get_score(importance_type='gain')
# xgboost用f0-f9索引，映射回特征名
importance_df = pd.DataFrame({
    'feature_idx': [int(k[1:]) for k in importance_dict.keys()],
    'importance': list(importance_dict.values())
})
importance_df['feature'] = importance_df['feature_idx'].apply(lambda i: feature_names[i])
importance_df = importance_df.sort_values('importance', ascending=False)

print("\n  特征重要性 (Gain):")
for _, row in importance_df.iterrows():
    print(f"    {row['feature']:20s}: {row['importance']:.4f}")

# ═══════════════════════════════════════════════════════════════════════════
# 4. SHAP解释
# ═══════════════════════════════════════════════════════════════════════════
print("\n【步骤4】SHAP解释...")

# 计算SHAP值（使用TreeExplainer — 更快，精确）
print("  计算SHAP值 (TreeExplainer)...")
explainer = shap.TreeExplainer(full_model)
shap_values = explainer.shap_values(X)

# 计算每个特征的mean |SHAP|
mean_abs_shap = np.abs(shap_values).mean(axis=0)
shap_importance = pd.DataFrame({
    'feature': feature_names,
    'mean_abs_shap': mean_abs_shap
}).sort_values('mean_abs_shap', ascending=False)
print(f"\n  SHAP 特征重要性 (mean |SHAP|):")
for _, row in shap_importance.iterrows():
    print(f"    {row['feature']:20s}: {row['mean_abs_shap']:.6f}")

# ─−− 图1: SHAP Summary Plot (beeswarm) ─−−
print("\n  生成图1: SHAP Summary Plot...")
fig, ax = plt.subplots(figsize=(10, 7))
shap.summary_plot(
    shap_values, X, feature_names=feature_names,
    show=False, max_display=10, plot_size=(10, 6)
)
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, 'shap_summary_plot.png'), dpi=300, bbox_inches='tight')
plt.close()
print("    → figures/shap_summary_plot.png")

# ─−− 图2: SHAP Bar Plot ─−−
print("\n  生成图2: SHAP Bar Plot...")
fig, ax = plt.subplots(figsize=(10, 6))
shap.summary_plot(
    shap_values, X, feature_names=feature_names,
    show=False, plot_type='bar', max_display=10, plot_size=(10, 5)
)
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, 'shap_bar_plot.png'), dpi=300, bbox_inches='tight')
plt.close()
print("    → figures/shap_bar_plot.png")

# ─−− 图3: SHAP Dependence Plot for Timing ─−−
# 关键图：展示运动时机对预测MetS的边际贡献
# 使用timing one-hot编码中的核心变量 + 组合值
print("\n  生成图3: SHAP Dependence Plot (运动时机)...")

# 方法：将5个timing SHAP值合并为1个"total timing SHAP"
# 并绘制与timing类别的关系
timing_shap_indices = [5, 6, 7, 8, 9]  # 对应 feature_names 中的 Timing_*

# 计算每个样本在timing相关的SHAP贡献总和
timing_total_shap = shap_values[:, timing_shap_indices].sum(axis=1)

# 图3a: 按dominant_timing类别，展示timing SHAP总值的分布
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 左侧: 小提琴图
timing_labels = df['dominant_timing'].values
timing_colors = {'Morning': '#4C72B0', 'Noon': '#DD8452', 'Afternoon': '#55A868',
                 'Evening': '#C44E52', 'Mixed': '#937860'}
timing_order_plot = ['Morning', 'Noon', 'Afternoon', 'Evening', 'Mixed']

data_list = []
for t in timing_order_plot:
    mask = timing_labels == t
    data_list.append(timing_total_shap[mask])

bp = axes[0].boxplot(data_list, labels=timing_order_plot, patch_artist=True,
                      widths=0.6, showmeans=True,
                      meanprops=dict(marker='D', markerfacecolor='red', markersize=6))
for patch, color in zip(bp['boxes'], [timing_colors[t] for t in timing_order_plot]):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
axes[0].set_title('Total Timing SHAP Value by Exercise Timing Group', fontsize=14, fontweight='bold')
axes[0].set_ylabel('SHAP Value (Total Timing Contribution)', fontsize=12)
axes[0].set_xlabel('Dominant Exercise Timing', fontsize=12)
axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
axes[0].tick_params(axis='x', rotation=0)

# 右侧: 散点图 - SHAP值 vs MVPA volume, 按timing着色
for t in timing_order_plot:
    mask = timing_labels == t
    axes[1].scatter(np.log1p(df.loc[mask, 'Total_MVPA']), timing_total_shap[mask],
                    c=timing_colors[t], label=t, alpha=0.4, s=15, edgecolors='none')
axes[1].set_title('Timing SHAP vs MVPA Volume', fontsize=14, fontweight='bold')
axes[1].set_xlabel('log(MVPA Volume + 1)', fontsize=12)
axes[1].set_ylabel('Timing SHAP Value (Total)', fontsize=12)
axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
axes[1].legend(fontsize=11)

plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, 'shap_timing_dependence.png'), dpi=300, bbox_inches='tight')
plt.close()
print("    → figures/shap_timing_dependence.png")

# ─−− 图3b: 各timing类别的SHAP细分（每个one-hot变量分别看） ───
fig, ax = plt.subplots(figsize=(12, 6))
x_pos = np.arange(5)
width = 0.15

for i, t_name in enumerate(['Morning', 'Noon', 'Afternoon', 'Evening', 'Mixed']):
    idx = timing_shap_indices[i]
    means = []
    for t in timing_order_plot:
        mask = timing_labels == t
        means.append(shap_values[mask, idx].mean())
    ax.bar(x_pos + i*width, means, width, label=t_name,
           color=timing_colors[t_name], alpha=0.8)

ax.set_xticks(x_pos + width*2)
ax.set_xticklabels(timing_order_plot)
ax.set_xlabel('Actual Exercise Timing Group', fontsize=12)
ax.set_ylabel('Mean SHAP Value', fontsize=12)
ax.set_title('Decomposition: Per-Timing-Variable SHAP by Actual Timing Group', fontsize=14, fontweight='bold')
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.legend(title='SHAP Variable', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, 'shap_timing_decomposition.png'), dpi=300, bbox_inches='tight')
plt.close()
print("    → figures/shap_timing_decomposition.png")

# ─−− 图4: ROC曲线 (5折CV汇总) ───
print("\n  生成图4: ROC曲线...")
fig, ax = plt.subplots(figsize=(8, 7))

fpr, tpr, thresholds = roc_curve(all_y_true, all_y_prob)
auc_mean = np.mean(cv_scores['auc'])
ax.plot(fpr, tpr, 'b-', linewidth=2,
        label=f'XGBoost (AUC={auc_mean:.3f} ± {np.std(cv_scores["auc"]):.3f})')
ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Random (AUC=0.5)')
ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12)
ax.set_title('XGBoost ROC Curve - MetS Prediction', fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='lower right')
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])

plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, 'roc_curve.png'), dpi=300, bbox_inches='tight')
plt.close()
print("    → figures/roc_curve.png")

# ═══════════════════════════════════════════════════════════════════════════
# 5. 对比: XGBoost SHAP vs Logistic OR
# ═══════════════════════════════════════════════════════════════════════════
print("\n【步骤5】对比: XGBoost SHAP vs Logistic回归 OR...")

import statsmodels.api as sm

# 用相同数据跑Logistic回归（Model 3: 完全调整）
# 填充可能的NaN (BMI缺失已在前面填充)
df_analysis['bmi_centered'] = df_analysis['bmi_centered'].fillna(0)
X_logit = sm.add_constant(df_analysis[[
    'Timing_Noon', 'Timing_Afternoon', 'Timing_Evening', 'Timing_Mixed',
    'age_centered', 'RIAGENDR', 'race_black', 'race_hispanic',
    'bmi_centered', 'mvpa_log'
]].astype(float))
y_logit = df_analysis['has_mets'].astype(float)
logit_model = sm.Logit(y_logit, X_logit).fit(disp=0)

print("\n  Logistic回归结果 (Model 3 - 完全调整):")
logit_results = []
for var in X_logit.columns[1:]:
    coef = logit_model.params[var]
    or_val = np.exp(coef)
    ci_low = np.exp(coef - 1.96 * logit_model.bse[var])
    ci_high = np.exp(coef + 1.96 * logit_model.bse[var])
    pval = logit_model.pvalues[var]
    print(f"    {var:25s}: OR={or_val:.4f} ({ci_low:.4f}-{ci_high:.4f}), p={pval:.4f}")
    logit_results.append({
        'variable': var, 'OR': or_val, 'CI_low': ci_low, 'CI_high': ci_high, 'p': pval
    })

# SHAP与Logistic OR对比表
print("\n  ─── SHAP vs Logistic OR 对比 ───")
print("  Timings (参考组=Morning):")
print(f"  {'Timing':20s} {'Logistic OR':15s} {'OR_CI':25s} {'SHAP_mean':15s} {'方向一致?':10s}")

# 计算每个timing组的平均SHAP值（相对Morning）
all_timing_shap_mean = {}
for i, t_idx in enumerate(timing_shap_indices):
    t_name = feature_names[t_idx].replace('Timing_', '')
    all_timing_shap_mean[t_name] = np.mean(shap_values[:, t_idx])

for logit_var in ['Timing_Noon', 'Timing_Afternoon', 'Timing_Evening', 'Timing_Mixed']:
    t_name = logit_var.replace('Timing_', '')
    lr_or = np.exp(logit_model.params[logit_var])
    lr_ci_low = np.exp(logit_model.params[logit_var] - 1.96*logit_model.bse[logit_var])
    lr_ci_high = np.exp(logit_model.params[logit_var] + 1.96*logit_model.bse[logit_var])
    
    shap_mean = all_timing_shap_mean.get(t_name, 0)
    
    # 方向一致性: Logistic OR>1 & SHAP mean>0 OR OR<1 & SHAP mean<0
    direction_match = (lr_or > 1 and shap_mean > 0) or (lr_or < 1 and shap_mean < 0)
    direction_symbol = '✅' if direction_match else '⚠️'
    
    print(f"  {t_name:20s} {lr_or:<15.4f} ({lr_ci_low:.4f}-{lr_ci_high:.4f})  "
          f"{shap_mean:<+15.6f} {direction_symbol:10s}")

# ═══════════════════════════════════════════════════════════════════════════
# 6. 保存结果
# ═══════════════════════════════════════════════════════════════════════════
print("\n【步骤6】保存结果...")

results = {
    'data_info': {
        'n_samples': len(df),
        'n_mets_positive': int(y.sum()),
        'mets_prevalence': float(y.mean()),
        'n_features': X.shape[1],
        'features': feature_names,
        'timing_distribution': df['dominant_timing'].value_counts().to_dict(),
    },
    'xgb_params': xgb_params,
    'cv_results': {
        'auc_mean': float(np.mean(cv_scores['auc'])),
        'auc_std': float(np.std(cv_scores['auc'])),
        'precision_mean': float(np.mean(cv_scores['precision'])),
        'precision_std': float(np.std(cv_scores['precision'])),
        'recall_mean': float(np.mean(cv_scores['recall'])),
        'recall_std': float(np.std(cv_scores['recall'])),
        'f1_mean': float(np.mean(cv_scores['f1'])),
        'f1_std': float(np.std(cv_scores['f1'])),
        'accuracy_mean': float(np.mean(cv_scores['accuracy'])),
        'accuracy_std': float(np.std(cv_scores['accuracy'])),
    },
    'confusion_matrix': cm.tolist(),
    'feature_importance_xgb': importance_df.to_dict('records'),
    'feature_importance_shap': shap_importance.to_dict('records'),
    'shap_timing_by_group': {
        t: {
            'mean': float(np.mean(timing_total_shap[timing_labels == t])),
            'std': float(np.std(timing_total_shap[timing_labels == t])),
            'n': int(np.sum(timing_labels == t))
        }
        for t in timing_order_plot
    },
    'logistic_comparison': {
        'model_summary': str(logit_model.summary()),
        'results': logit_results,
    }
}

# 保存JSON结果
results_json_path = os.path.join(OUTPUT_DIR, 'xgboost_results.json')
with open(results_json_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False, default=str)
print(f"  → {results_json_path}")

print("\n" + "=" * 70)
print("XGBoost + SHAP 分析完成！")
print(f"所有图保存至: {FIGURE_DIR}")
print("=" * 70)
print("\n要点解读:")
print(f"  1. XGBoost AUC = {np.mean(cv_scores['auc']):.3f} ± {np.std(cv_scores['auc']):.3f}")
print(f"  2. SHAP top特征: {shap_importance.iloc[0]['feature']}, {shap_importance.iloc[1]['feature']}")
print(f"  3. 运动时机(Mixed/Evening/Noon/Afternoon vs Morning)对预测MetS的边际贡献")
print(f"    请查看: figures/shap_timing_dependence.png")
print(f"  4. SHAP vs Logistic OR方向一致性: 见上述对比表")
