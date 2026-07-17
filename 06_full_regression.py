#!/usr/bin/env python3
"""
运动时机与代谢健康研究 - 完整回归分析
太阳·心火·运动 | 2026-04-19
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

output_dir = '/root/.openclaw/workspace/projects/exercise-timing-paper/03_analysis'
df = pd.read_csv(f'{output_dir}/final_analysis_data.csv')

print("="*70)
print("运动时机与代谢综合征 - 多因素Logistic回归分析")
print(f"分析样本: {len(df)}")
print("="*70)

# 创建哑变量（Morning为参考）
for t in ['Morning', 'Noon', 'Afternoon', 'Evening', 'Mixed']:
    df[f'timing_{t}'] = (df['dominant_timing'] == t).astype(int)

# 协变量
df['age_10'] = df['RIDAGEYR'] / 10
df['bmi_5'] = df['BMXBMI'].fillna(df['BMXBMI'].median()) / 5
df['mvpa_log'] = np.log1p(df['Total_MVPA'])
df['male'] = df['RIAGENDR'].values
df['race_black'] = (df['RIDRETH1'] == 4).astype(int)
df['race_hispanic'] = ((df['RIDRETH1']==1)|(df['RIDRETH1']==2)).astype(int)
df['race_other'] = 1 - df['race_black'] - df['race_hispanic'] - (df['RIDRETH1']==3).astype(int)

y = df['has_mets'].astype(float)

def run_logistic(y, X_vars, data):
    X = data[X_vars].astype(float)
    X = sm.add_constant(X)
    model = sm.Logit(y, X).fit(disp=0, maxiter=100)
    return model

def get_or(model, var):
    c = model.params[var]
    s = model.bse[var]
    return np.exp(c), np.exp(c-1.96*s), np.exp(c+1.96*s), model.pvalues[var]

# Model 1: 未调整
print("\n【Model 1】未调整")
vars1 = ['timing_Noon', 'timing_Afternoon', 'timing_Evening', 'timing_Mixed']
m1 = run_logistic(y, vars1, df)
print(f"{'Group':<12} {'OR':>6} {'95%CI':>16} {'P':>8}")
print("-"*45)
for t in ['Noon', 'Afternoon', 'Evening', 'Mixed']:
    o, lo, hi, p = get_or(m1, f'timing_{t}')
    print(f"{t:<12} {o:.3f} ({lo:.3f}-{hi:.3f}) {p:.3f}")

# Model 2: 调整人口学
print("\n【Model 2】调整年龄、性别、种族")
vars2 = vars1 + ['age_10', 'male', 'race_black', 'race_hispanic', 'race_other']
m2 = run_logistic(y, vars2, df)
print(f"{'Group':<12} {'OR':>6} {'95%CI':>16} {'P':>8}")
print("-"*45)
for t in ['Noon', 'Afternoon', 'Evening', 'Mixed']:
    o, lo, hi, p = get_or(m2, f'timing_{t}')
    print(f"{t:<12} {o:.3f} ({lo:.3f}-{hi:.3f}) {p:.3f}")

# Model 3: 完全调整
print("\n【Model 3】完全调整（+BMI+MVPA）")
vars3 = vars2 + ['bmi_5', 'mvpa_log']
m3 = run_logistic(y, vars3, df)
print(f"{'Group':<12} {'OR':>6} {'95%CI':>16} {'P':>8}")
print("-"*45)
for t in ['Noon', 'Afternoon', 'Evening', 'Mixed']:
    o, lo, hi, p = get_or(m3, f'timing_{t}')
    print(f"{t:<12} {o:.3f} ({lo:.3f}-{hi:.3f}) {p:.3f}")

# 汇总回归结果
reg_results = []
for mn, m in [('Model 1 (Unadjusted)', m1), ('Model 2 (Demographics)', m2), ('Model 3 (Fully Adjusted)', m3)]:
    for t in ['Noon', 'Afternoon', 'Evening', 'Mixed']:
        o, lo, hi, p = get_or(m, f'timing_{t}')
        reg_results.append({'Model': mn, 'Timing': t, 'OR': o, 'CI_Lower': lo, 'CI_Upper': hi, 'P': p})

pd.DataFrame(reg_results).to_csv(f'{output_dir}/regression_results.csv', index=False)

# 亚组分析
print("\n" + "="*70)
print("亚组分析")
print("="*70)

sub_results = []
def subgroup_analysis(sub_df, name):
    if len(sub_df) < 100: return
    mets_r = sub_df['has_mets'].mean()
    try:
        m = run_logistic(sub_df['has_mets'].astype(float), vars3, sub_df)
        for t in ['Noon', 'Afternoon', 'Evening', 'Mixed']:
            o, lo, hi, p = get_or(m, f'timing_{t}')
            sub_results.append({
                'Subgroup': name, 'N': len(sub_df), 'MetS%': f"{mets_r*100:.1f}",
                'Timing': t, 'OR': o, 'CI_Lower': lo, 'CI_Upper': hi, 'P': p
            })
    except: pass

# 性别
for g, n in [(1, 'Male'), (2, 'Female')]:
    sub = df[df['RIAGENDR']==g]
    subgroup_analysis(sub, n)

# 年龄
for lo, hi, n in [(18,45,'<45'), (45,65,'45-64'), (65,100,'≥65')]:
    sub = df[(df['RIDAGEYR']>=lo)&(df['RIDAGEYR']<hi)]
    subgroup_analysis(sub, n)

# BMI
for lo, hi, n in [(0,25,'Normal'), (25,30,'Overweight'), (30,100,'Obese')]:
    sub = df[(df['BMXBMI']>=lo)&(df['BMXBMI']<hi)]
    subgroup_analysis(sub, n)

# 种族
for r, n in [(3,'Non-Hispanic White'), (4,'Non-Hispanic Black')]:
    sub = df[df['RIDRETH1']==r]
    subgroup_analysis(sub, n)
sub = df[((df['RIDRETH1']==1)|(df['RIDRETH1']==2))]
subgroup_analysis(sub, 'Hispanic')

# 输出亚组结果
for sg in sub_results:
    print(f"{sg['Subgroup']:<22} {sg['Timing']:<12} n={sg['N']:<5} MetS={sg['MetS%']:<6} OR={sg['OR']:.2f} ({sg['CI_Lower']:.2f}-{sg['CI_Upper']:.2f}) p={sg['P']:.3f}")

pd.DataFrame(sub_results).to_csv(f'{output_dir}/subgroup_analysis.csv', index=False)

# 各MetS组分分析
print("\n" + "="*70)
print("各MetS组分分析")
print("="*70)

df['met_waist'] = np.where(df['RIAGENDR']==1, (df['BMXWAIST']>102).astype(int), (df['BMXWAIST']>88).astype(int))
df['met_tg'] = (df['LBXTR']>=150).astype(int)
df['met_hdl'] = np.where(df['RIAGENDR']==1, (df['HDL']<40).astype(int), (df['HDL']<50).astype(int))
df['met_bp'] = ((df['systolic']>=130)|(df['diastolic']>=85)).astype(int)
df['met_glu'] = (df['LBXGLU']>=100).astype(int)

comp_results = []
for comp, name in [('met_waist','Abdominal Obesity'), ('met_tg','Elevated TG'), 
                    ('met_hdl','Low HDL'), ('met_bp','Elevated BP'), ('met_glu','Elevated Glucose')]:
    print(f"\n{name}:")
    try:
        m = run_logistic(df[comp].astype(float), vars3, df)
        for t in ['Noon', 'Afternoon', 'Evening', 'Mixed']:
            o, lo, hi, p = get_or(m, f'timing_{t}')
            sig = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else ""
            print(f"  {t:<12} OR={o:.3f} ({lo:.3f}-{hi:.3f}) p={p:.3f} {sig}")
            comp_results.append({'Component': name, 'Timing': t, 'OR': o, 'CI_Lower': lo, 'CI_Upper': hi, 'P': p})
    except Exception as e:
        print(f"  Error: {e}")

pd.DataFrame(comp_results).to_csv(f'{output_dir}/component_analysis.csv', index=False)

# Table 1
print("\n" + "="*70)
print("Table 1: 基线特征")
print("="*70)

table1 = []
for var, label in [('RIDAGEYR','Age, years'), ('RIAGENDR','Male, %'), ('BMXBMI','BMI, kg/m²'),
                    ('BMXWAIST','Waist, cm'), ('systolic','SBP, mmHg'), ('diastolic','DBP, mmHg'),
                    ('LBXGLU','FPG, mg/dL'), ('LBXTR','TG, mg/dL'), ('HDL','HDL-C, mg/dL'),
                    ('Total_MVPA','MVPA, min/wk')]:
    row = {'Variable': label}
    for tg in ['Morning','Noon','Afternoon','Evening','Mixed']:
        sub = df[df['dominant_timing']==tg][var]
        if var == 'RIAGENDR':
            row[tg] = f"{sub.mean()*100:.1f}"
        else:
            row[tg] = f"{sub.mean():.1f}±{sub.std():.1f}"
    table1.append(row)

t1 = pd.DataFrame(table1)
t1.to_csv(f'{output_dir}/table1_baseline.csv', index=False)

# MetS患病率行
mets_row = {'Variable': 'MetS, %'}
for tg in ['Morning','Noon','Afternoon','Evening','Mixed']:
    mets_row[tg] = f"{df[df['dominant_timing']==tg]['has_mets'].mean()*100:.1f}"
table1.append(mets_row)

print(pd.DataFrame(table1).to_string(index=False))

print("\n" + "="*70)
print("所有分析结果已保存！")
print("="*70)
