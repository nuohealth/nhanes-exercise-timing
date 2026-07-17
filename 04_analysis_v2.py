#!/usr/bin/env python3
"""
运动时机与代谢健康研究 - 完整分析v2
太阳·心火·运动
2026-04-19

使用NHANES 2011-2014加速度计数据
基于MIMS活动量定义运动时机分组
"""

import pandas as pd
import numpy as np
import pyreadstat
import os
import warnings
warnings.filterwarnings('ignore')

data_dir = '/root/.openclaw/workspace/projects/exercise-timing-paper/02_data/nhanes_raw'
output_dir = '/root/.openclaw/workspace/projects/exercise-timing-paper/03_analysis'

print("="*70)
print("运动时机与代谢综合征研究 - 完整分析v2")
print("="*70)

def parse_time_to_hour(time_val):
    if pd.isna(time_val):
        return None
    try:
        if isinstance(time_val, (int, float)):
            time_str = str(int(time_val)).zfill(6)
            return int(time_str[:2])
        else:
            parts = str(time_val).split(':')
            return int(parts[0])
    except:
        return None

def process_pax_timing(cycle_suffix, cycle_name):
    """处理加速度计数据，计算运动时机"""
    print(f"\n【处理 {cycle_name}】")
    
    # 读取数据
    paxhr, _ = pyreadstat.read_xport(os.path.join(data_dir, f'PAXHR_{cycle_suffix}.XPT'))
    paxhd, _ = pyreadstat.read_xport(os.path.join(data_dir, f'PAXHD_{cycle_suffix}.XPT'))
    
    # 解析开始时间
    paxhd['start_hour'] = paxhd['PAXFTIME'].apply(parse_time_to_hour)
    
    # 只保留完整小时的记录
    paxhr_60 = paxhr[paxhr['PAXTMH'] == 60].copy()
    
    # 合并开始时间
    merged = paxhr_60.merge(
        paxhd[['SEQN', 'start_hour']], 
        on='SEQN', how='left'
    )
    
    # 计算实际小时（24小时制）
    merged['actual_hour'] = (merged['start_hour'] + merged['PAXSSNHP'] / (80 * 3600)) % 24
    
    # 定义时间窗
    def get_window(h):
        if 5 <= h < 11: return 'Morning'
        elif 11 <= h < 14: return 'Noon'
        elif 14 <= h < 17: return 'Afternoon'
        elif 17 <= h < 21: return 'Evening'
        else: return 'Night'
    
    merged['time_window'] = merged['actual_hour'].apply(get_window)
    
    # 计算活动量指标
    # 使用清醒分钟的活动量（MIMS/min）作为活动强度指标
    merged['mims_per_min'] = merged['PAXMTSH'] / merged['PAXWWMH'].replace(0, np.nan)
    
    # MVPA阈值：使用MIMS > 25/min（基于数据分布，约90th percentile）
    # 参考：John et al. 2019 MIMS paper
    merged['is_mvpa'] = (merged['mims_per_min'] > 25).astype(int)
    merged['mvpa_min'] = merged['is_mvpa'] * merged['PAXWWMH']
    
    # 按参与者和时间窗汇总
    summary = merged.groupby(['SEQN', 'time_window']).agg({
        'mvpa_min': 'sum',
        'PAXWWMH': 'sum',
        'PAXMTSH': 'sum'
    }).reset_index()
    
    pivot = summary.pivot(index='SEQN', columns='time_window', values='mvpa_min').fillna(0)
    for w in ['Morning', 'Noon', 'Afternoon', 'Evening', 'Night']:
        if w not in pivot.columns:
            pivot[w] = 0
    
    pivot['Total_MVPA'] = pivot[['Morning', 'Noon', 'Afternoon', 'Evening']].sum(axis=1)
    
    def get_dominant(row):
        if row['Total_MVPA'] < 30:
            return 'Inactive'
        windows = ['Morning', 'Noon', 'Afternoon', 'Evening']
        max_w = max(windows, key=lambda w: row[w])
        if row[max_w] / row['Total_MVPA'] >= 0.3:
            return max_w
        return 'Mixed'
    
    pivot['dominant_timing'] = pivot.apply(get_dominant, axis=1)
    pivot = pivot.reset_index()
    
    # 也保存原始活动量用于其他分析
    summary_wear = merged.groupby(['SEQN', 'time_window']).agg({
        'PAXWWMH': 'sum',
        'PAXMTSH': 'sum'
    }).reset_index()
    wear_pivot = summary_wear.pivot(index='SEQN', columns='time_window', values=['PAXWWMH', 'PAXMTSH']).fillna(0)
    wear_pivot.columns = [f'{col}_{w}' for col, w in wear_pivot.columns]
    wear_pivot = wear_pivot.reset_index()
    
    result = pivot.merge(wear_pivot, on='SEQN', how='left')
    result['cycle'] = cycle_name
    
    print(f"  参与者数: {len(result)}")
    print(f"  主导运动时机分布:")
    print(result['dominant_timing'].value_counts())
    
    return result

# 处理两个周期
mvpa_g = process_pax_timing('G', '2011-2012')
mvpa_h = process_pax_timing('H', '2013-2014')

mvpa_all = pd.concat([mvpa_g, mvpa_h], ignore_index=True)
print(f"\n【汇总】总参与者数: {len(mvpa_all)}")
print(mvpa_all['dominant_timing'].value_counts())

# 读取其他数据
def read_clinical(cycle_suffix, cycle_name):
    demo, _ = pyreadstat.read_xport(os.path.join(data_dir, f'DEMO_{cycle_suffix}.XPT'))
    demo = demo[['SEQN', 'RIAGENDR', 'RIDAGEYR', 'RIDRETH1', 'DMDEDUC2', 'SDMVPSU', 'SDMVSTRA']].copy()
    
    bmx, _ = pyreadstat.read_xport(os.path.join(data_dir, f'BMX_{cycle_suffix}.XPT'))
    bmx = bmx[['SEQN', 'BMXBMI', 'BMXWAIST']].copy()
    
    df = demo.merge(bmx, on='SEQN', how='inner')
    
    bpx, _ = pyreadstat.read_xport(os.path.join(data_dir, f'BPX_{cycle_suffix}.XPT'))
    bpx['systolic'] = bpx[['BPXSY1', 'BPXSY2', 'BPXSY3', 'BPXSY4']].mean(axis=1, skipna=True)
    bpx['diastolic'] = bpx[['BPXDI1', 'BPXDI2', 'BPXDI3', 'BPXDI4']].mean(axis=1, skipna=True)
    df = df.merge(bpx[['SEQN', 'systolic', 'diastolic']], on='SEQN', how='left')
    
    try:
        glu, _ = pyreadstat.read_xport(os.path.join(data_dir, f'GLU_{cycle_suffix}.XPT'))
        df = df.merge(glu[['SEQN', 'LBXGLU']], on='SEQN', how='left')
    except: df['LBXGLU'] = np.nan
    
    try:
        trig, _ = pyreadstat.read_xport(os.path.join(data_dir, f'TRIGLY_{cycle_suffix}.XPT'))
        df = df.merge(trig[['SEQN', 'LBXTR']], on='SEQN', how='left')
    except: df['LBXTR'] = np.nan
    
    try:
        hdl, _ = pyreadstat.read_xport(os.path.join(data_dir, f'HDL_{cycle_suffix}.XPT'))
        col = 'LBDHDD' if 'LBDHDD' in hdl.columns else [c for c in hdl.columns if 'HDD' in c][0]
        df = df.merge(hdl[['SEQN', col]].rename(columns={col: 'HDL'}), on='SEQN', how='left')
    except: df['HDL'] = np.nan
    
    df['cycle'] = cycle_name
    return df

print("\n【读取临床数据】")
df_g = read_clinical('G', '2011-2012')
df_h = read_clinical('H', '2013-2014')
df_all = pd.concat([df_g, df_h], ignore_index=True)
print(f"临床数据: {len(df_all)}")

# 合并
analysis = mvpa_all.merge(df_all, on=['SEQN', 'cycle'], how='inner')
analysis = analysis[analysis['RIDAGEYR'] >= 18].copy()
print(f"合并后成人样本: {len(analysis)}")

# 定义MetS
print("\n【定义MetS】")

def calc_mets(row):
    c = 0
    # 腰围
    if row['RIAGENDR'] == 1:
        c += 1 if row['BMXWAIST'] > 102 else 0
    else:
        c += 1 if row['BMXWAIST'] > 88 else 0
    # TG
    c += 1 if row['LBXTR'] >= 150 else 0
    # HDL
    if row['RIAGENDR'] == 1:
        c += 1 if row['HDL'] < 40 else 0
    else:
        c += 1 if row['HDL'] < 50 else 0
    # BP
    c += 1 if (row['systolic'] >= 130 or row['diastolic'] >= 85) else 0
    # FPG
    c += 1 if row['LBXGLU'] >= 100 else 0
    return c

analysis['mets_count'] = analysis.apply(calc_mets, axis=1)
analysis['has_mets'] = (analysis['mets_count'] >= 3).astype(int)

valid = analysis['mets_count'].notna()
print(f"  有效MetS数据: {valid.sum()}")
print(f"  MetS患病率: {analysis.loc[valid, 'has_mets'].mean()*100:.1f}%")

# 筛选活跃参与者
final = analysis[
    (analysis['dominant_timing'] != 'Inactive') &
    (analysis['mets_count'].notna())
].copy()

print(f"\n【最终分析样本: {len(final)}】")
print(f"\n运动时机分布:")
print(final['dominant_timing'].value_counts())
print(f"\nMetS患病率（按运动时机）:")
mets_rate = final.groupby('dominant_timing')['has_mets'].agg(['mean', 'count', 'sum'])
mets_rate.columns = ['患病率', '总数', 'MetS人数']
mets_rate['患病率%'] = mets_rate['患病率'] * 100
print(mets_rate[['患病率%', '总数', 'MetS人数']])

# 基线特征
print("\n【基线特征 - 按运动时机分组】")
vars_info = {
    'RIDAGEYR': '年龄(岁)',
    'RIAGENDR': '男性(%)',
    'BMXBMI': 'BMI(kg/m²)',
    'BMXWAIST': '腰围(cm)',
    'systolic': '收缩压(mmHg)',
    'diastolic': '舒张压(mmHg)',
    'LBXGLU': '空腹血糖(mg/dL)',
    'LBXTR': '甘油三酯(mg/dL)',
    'HDL': 'HDL胆固醇(mg/dL)',
    'Total_MVPA': '总MVPA(min)',
}

for var, label in vars_info.items():
    print(f"\n{label}:")
    stats = final.groupby('dominant_timing')[var].agg(['mean', 'std'])
    if var == 'RIAGENDR':
        stats['mean'] = stats['mean'] * 100
    print(stats.round(1))

# 保存
final.to_csv(os.path.join(output_dir, 'final_analysis_data.csv'), index=False)
print(f"\n数据已保存至 {output_dir}/final_analysis_data.csv")
print("="*70)
print("数据准备完成！下一步：统计回归分析")
print("="*70)
