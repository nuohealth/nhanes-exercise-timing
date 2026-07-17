#!/usr/bin/env python3
"""
太阳·心火 — 重新生成真实的Figure_S1和Figure_S2

Figure_S1: 用R svyglm真实OR数据替换旧sklearn版本
Figure_S2: 用真实XGBoost 5-fold CV OOF预测结果生成ROC+校准曲线

所有其他图(Figure_1, Figure_2, Figure_S3, Figure_S4)保持不变。
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import os
import warnings
import json
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve
import xgboost as xgb

# ── 路径 ──
FIGS_DIR = '/root/.openclaw/workspace/projects/exercise-timing-paper/figures'
REVIEW_DIR = '/root/.openclaw/workspace/projects/exercise-timing-paper/review/round1'
DATA_PATH = '/root/.openclaw/workspace/projects/exercise-timing-paper/03_analysis/final_analysis_data.csv'
os.makedirs(FIGS_DIR, exist_ok=True)

# ── 输出到PeerJ目录 ──
PEERJ_DIR = '/APapers/PeerJ-Submission'
os.makedirs(PEERJ_DIR, exist_ok=True)

# ── 样式 ──
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica', 'sans-serif'],
    'font.size': 10,
    'axes.linewidth': 1.0,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.15,
})


# ============================================================
# FIGURE S1 (替换): 用R svyglm的真实加权OR替代旧版sklearn数据
# ============================================================
def make_figureS1():
    """用R svyglm结果（r_svyglm_results.csv）生成加权vs未加权OR对比"""
    
    # ── 数据 ──
    # 数据来源1: R svyglm Model 1 unadjusted (Evening OR=0.761)
    # 数据来源2: R svyglm Model 3 fully adjusted (Evening OR=1.092)
    # 
    # 对比展示:
    #   1. Unadjusted (未加权) — 来自Table 2 Model 1 Evening: OR=0.78 (0.65-0.93)
    #   2. Survey-weighted (unadjusted) — R svyglm Model 1 Evening: OR=0.761 (0.593-0.976)
    #   3. Fully adjusted (未加权) — Table 2 Model 3 Evening: OR=1.13 (0.93-1.37)
    #   4. Survey-weighted (fully adjusted) — R svyglm Model 3 Evening: OR=1.092 (0.842-1.414)
    
    WEIGHTED_DATA = [
        {
            'label': 'Unadjusted\n(unweighted)',
            'or': 0.78,
            'ci_low': 0.65,
            'ci_high': 0.93,
        },
        {
            'label': 'Unadjusted\n(weighted\nsvyglm)',
            'or': 0.761,
            'ci_low': 0.593,
            'ci_high': 0.976,
        },
        {
            'label': 'Fully adjusted\n(unweighted)',
            'or': 1.13,
            'ci_low': 0.93,
            'ci_high': 1.37,
        },
        {
            'label': 'Fully adjusted\n(weighted\nsvyglm)',
            'or': 1.092,
            'ci_low': 0.842,
            'ci_high': 1.414,
        },
    ]
    
    labels = [d['label'] for d in WEIGHTED_DATA]
    or_vals = [d['or'] for d in WEIGHTED_DATA]
    ci_low  = [d['ci_low'] for d in WEIGHTED_DATA]
    ci_high = [d['ci_high'] for d in WEIGHTED_DATA]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    y_pos = np.arange(len(labels))

    # Color scheme: unweighted=blue, weighted=orange
    colors = ['#4DBBD5', '#E64B35', '#4DBBD5', '#E64B35']

    for i in range(len(labels)):
        ax.plot([ci_low[i], ci_high[i]], [y_pos[i], y_pos[i]],
                color=colors[i], lw=3, zorder=4)
    ax.scatter(or_vals, y_pos, s=150, marker='D', color=colors,
               zorder=6, edgecolors='white', linewidth=0.8)

    ax.axvline(x=1.0, color='#888', linestyle='--', linewidth=1.0, alpha=0.6, zorder=0)
    ax.axvline(x=0.78, color='#4DBBD5', linestyle=':', linewidth=0.8, alpha=0.3, zorder=0)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel('Odds Ratio (Evening vs Morning)', fontsize=11, fontweight='bold')
    ax.set_xlim(0.35, 1.75)
    ax.set_ylim(-0.5, len(labels) - 0.5)

    for i in range(len(labels)):
        ax.text(ci_high[i] + 0.03, y_pos[i],
                f"OR={or_vals[i]:.3f}\n({ci_low[i]:.3f}-{ci_high[i]:.3f})",
                fontsize=7.5, va='center', ha='left', color='#333')

    ax.set_xticks(np.arange(0.4, 1.8, 0.2))
    ax.set_title('Figure S1. Survey-Weighted Sensitivity Analysis\n(Evening vs Morning, R svyglm with NHANES Design)',
                 fontsize=10, fontweight='bold', pad=10)

    legend_elements = [
        mpatches.Patch(facecolor='#4DBBD5', edgecolor='white', label='Unweighted', alpha=0.7),
        mpatches.Patch(facecolor='#E64B35', edgecolor='white', label='Survey-weighted (svyglm)', alpha=0.7),
    ]
    ax.legend(handles=legend_elements, fontsize=8.5, loc='lower right',
              framealpha=0.9, edgecolor='#ddd')

    ax.text(0.02, -0.5,
            'Survey-weighted models using R survey::svyglm() with NHANES PSU + strata (Taylor linearization).\n'
            'Unadjusted: no covariates. Fully adjusted: age, sex, race/ethnicity, BMI, education, log(MVPA).',
            fontsize=7, color='#555', transform=ax.transAxes)

    # Save
    fig.savefig(os.path.join(FIGS_DIR, 'figureS1_weighted_comparison.png'), dpi=300)
    fig.savefig(os.path.join(PEERJ_DIR, 'Figure_S1.png'), dpi=300)
    plt.close()
    print("✅ Figure S1 saved — REAL R svyglm OR values.")


# ============================================================
# FIGURE S2 (替换): 用真实XGBoost 5-fold CV OOF预测
# ============================================================
def make_figureS2():
    """重跑XGBoost 5-fold CV，获取真实OOF预测结果画ROC和校准曲线"""
    
    print("  加载数据...")
    df = pd.read_csv(DATA_PATH)
    
    # 特征编码（与xgboost_shap_analysis.py一致）
    feature_cols = ['Age', 'RIAGENDR', 'BMI', 'MVPA_Volume',
                    'race_black', 'race_hispanic', 'race_other']
    timing_dummies = ['Timing_Noon', 'Timing_Afternoon', 'Timing_Evening', 'Timing_Mixed']
    
    for td in timing_dummies:
        if td not in df.columns:
            df[td] = 0
    
    X = df[feature_cols + timing_dummies].values
    y = df['has_mets'].values
    
    neg_count = (y == 0).sum()
    pos_count = (y == 1).sum()
    scale_weight = neg_count / pos_count
    
    # ── 5-fold CV ──
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    all_y_true = []
    all_y_prob = []
    fold_aucs = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            scale_pos_weight=scale_weight,
            random_state=42,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='auc',
            use_label_encoder=False,
            verbosity=0,
        )
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_prob)
        
        all_y_true.extend(y_val)
        all_y_prob.extend(y_prob)
        fold_aucs.append(auc)
        print(f"  Fold {fold}: AUC={auc:.4f}")
    
    all_y_true = np.array(all_y_true)
    all_y_prob = np.array(all_y_prob)
    oof_auc = roc_auc_score(all_y_true, all_y_prob)
    mean_auc = np.mean(fold_aucs)
    std_auc = np.std(fold_aucs)
    
    print(f"  OOF AUC = {oof_auc:.4f}, Mean CV AUC = {mean_auc:.4f} ± {std_auc:.4f}")
    
    # ── ROC曲线数据 ──
    fpr, tpr, thresholds = roc_curve(all_y_true, all_y_prob)
    
    # ── 校准曲线数据 ──
    from sklearn.calibration import calibration_curve
    fraction_of_positives, mean_predicted_value = calibration_curve(
        all_y_true, all_y_prob, n_bins=10, strategy='uniform')
    
    # ── 绘图 ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.8))
    
    # Left: ROC
    ax1.plot(fpr, tpr, color='#3C5488', lw=2.5,
             label=f'XGBoost (AUC = {oof_auc:.3f})')
    ax1.plot([0, 1], [0, 1], '--', color='#888', lw=1, alpha=0.6, label='Random')
    ax1.fill_between(fpr, tpr, alpha=0.15, color='#3C5488')
    ax1.set_xlim(-0.02, 1.02)
    ax1.set_ylim(-0.02, 1.02)
    ax1.set_xlabel('1 - Specificity (False Positive Rate)', fontsize=9.5, fontweight='bold')
    ax1.set_ylabel('Sensitivity (True Positive Rate)', fontsize=9.5, fontweight='bold')
    ax1.set_title('ROC Curve', fontsize=11, fontweight='bold')
    ax1.legend(fontsize=8, loc='lower right', framealpha=0.9, edgecolor='#ddd')
    ax1.set_aspect('equal')
    
    # AUC annotation
    ax1.annotate(f'AUC = {oof_auc:.3f}',
                 xy=(0.55, 0.25), fontsize=10, fontweight='bold', color='#3C5488',
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#3C5488', alpha=0.8))
    
    # Right: Calibration
    ax2.plot([0, 1], [0, 1], '--', color='#888', lw=1.5, alpha=0.6, label='Perfect calibration')
    ax2.plot(mean_predicted_value, fraction_of_positives, 'o-', color='#E64B35', lw=2, markersize=6,
             markerfacecolor='white', markeredgecolor='#E64B35', markeredgewidth=1.5,
             label='XGBoost model')
    ax2.fill_between(mean_predicted_value, fraction_of_positives,
                     np.interp(mean_predicted_value, [0, 1], [0, 1]),
                     alpha=0.1, color='#E64B35')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_xlabel('Mean Predicted Probability', fontsize=9.5, fontweight='bold')
    ax2.set_ylabel('Observed Proportion', fontsize=9.5, fontweight='bold')
    ax2.set_title('Calibration Curve', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=8, loc='lower right', framealpha=0.9, edgecolor='#ddd')
    ax2.set_aspect('equal')
    
    fig.suptitle('Figure S2. XGBoost Model Performance\n(Unweighted, 5-fold Cross-Validation, OOF Predictions)',
                 fontsize=11, fontweight='bold', y=1.04)
    
    plt.tight_layout()
    fig.savefig(os.path.join(FIGS_DIR, 'figureS2_roc_calibration.png'), dpi=300)
    fig.savefig(os.path.join(PEERJ_DIR, 'Figure_S2.png'), dpi=300)
    plt.close()
    print(f"✅ Figure S2 saved — REAL XGBoost OOF AUC = {oof_auc:.3f}.")


# ============================================================
# 额外：确认Figure_S3 SHAP图的标签正确性
# ============================================================
def check_figureS3():
    """检查现有Figure_S3 SHAP图是否与正文Table 4一致"""
    shap_csv = os.path.join(REVIEW_DIR, 'survey_weights', 'shap_weighted_comparison.csv')
    if os.path.exists(shap_csv):
        df_shap = pd.read_csv(shap_csv)
        print("\n=== Figure S3 SHAP值验证 ===")
        print(df_shap.to_string(index=False))
        
        # 确认与正文Table 4的匹配关系
        expected = {
            'BMI': 0.712,
            'Age': 0.399,
            'log(MVPA)': 0.100,
            'Race': 0.073,
            'Education': 0.050,
        }
        for feat, exp_val in expected.items():
            # 在df中找对应行
            for _, row in df_shap.iterrows():
                if feat.lower() in row['Feature'].lower():
                    cvg = row['Unweighted_SHAP']
                    match = "✅" if abs(cvg - exp_val) < 0.01 else "⚠️"
                    print(f"  {match} {feat}: SHAP={cvg:.3f} vs expected={exp_val}")
    else:
        print("❌ 未找到shap_weighted_comparison.csv")


# ============================================================
# 确认余下的图都还在
# ============================================================
def verify_all_figures():
    """确认6张图全部存在"""
    expected_files = {
        'Figure_1.png': '流程图',
        'Figure_2.png': '森林图',
        'Figure_S1.png': '加权OR对比（替换）',
        'Figure_S2.png': 'ROC+校准曲线（替换）',
        'Figure_S3.png': 'SHAP图',
        'Figure_S4.png': '阈值敏感性分析',
    }
    
    print("\n=== 图文件验证 ===")
    for fname, desc in expected_files.items():
        path = os.path.join(PEERJ_DIR, fname)
        exists = os.path.exists(path)
        size = os.path.getsize(path) if exists else 0
        status = "✅" if exists else "❌"
        print(f"  {status} {fname}: {desc} ({size/1024:.0f} KB)")
    
    # 如果Figure_S3不存在，从figures目录找
    s3_path = os.path.join(PEERJ_DIR, 'Figure_S3.png')
    if not os.path.exists(s3_path):
        # 找
        import glob
        candidates = glob.glob(os.path.join(REVIEW_DIR, 'survey_weights', '*shap*unweighted*'))
        candidates += glob.glob(os.path.join(REVIEW_DIR, '*shap*unweighted*'))
        candidates += glob.glob(os.path.join(FIGS_DIR, '*shap*'))
        print(f"\n  寻找Figure_S3: {candidates}")


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("太阳·心火 — 重新生成真实Figure_S1和Figure_S2")
    print("=" * 60)
    print()
    
    print("[Figure S1] 替换加权OR数据为R svyglm真实值...")
    make_figureS1()
    print()
    
    print("[Figure S2] 重新运行XGBoost 5-fold CV生成真实ROC+校准曲线...")
    make_figureS2()
    print()
    
    check_figureS3()
    verify_all_figures()
    
    print()
    print("=" * 60)
    print("完成！新图已保存到:")
    print(f"  1. {PEERJ_DIR}/Figure_S1.png")
    print(f"  2. {PEERJ_DIR}/Figure_S2.png")
    print("=" * 60)
