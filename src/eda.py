"""
Hospital Readmission Analysis - Data Cleaning & EDA
Dataset: Diabetes 130-US Hospitals (UCI ML Repository)

Notes to self:
- Dataset has ~101k rows, 50 columns
- Target: predict 30-day readmission (readmitted column)
- '?' is used for missing values in the raw file, handled at load time
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
import os

warnings.filterwarnings('ignore')

# basic plot styling I prefer
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.color': '#e5e5e5',
    'font.size': 11,
})

os.makedirs('outputs', exist_ok=True)


# ── 1. Load data ─────────────────────────────────────────────────────────────

# the dataset uses '?' for nulls so I'm treating it as NaN directly
df = pd.read_csv('diabetic_data.csv', na_values=['?'])
print(f"Loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")


# ── 2. Quick audit before touching anything ───────────────────────────────────

# which columns have missing values and how bad is it?
missing = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
print("\nMissing values (% of rows):")
print(missing[missing > 0])

print("\nTarget distribution:")
print(df['readmitted'].value_counts())


# ── 3. Cleaning ───────────────────────────────────────────────────────────────

df_clean = df.copy()

# weight: 96.9% missing, not worth imputing - just drop it
# also dropping payer_code (40% missing, not relevant to readmission)
df_clean.drop(columns=['weight', 'payer_code'], inplace=True)

# medical_specialty has ~49% missing but the column is still useful where present
# filling with 'Unknown' so I don't lose those rows
df_clean['medical_specialty'].fillna('Unknown', inplace=True)
df_clean['race'].fillna('Unknown', inplace=True)

# one patient can have multiple encounters in this dataset
# keeping only the first encounter per patient to avoid leakage
before = len(df_clean)
df_clean = (df_clean
            .sort_values('encounter_id')
            .drop_duplicates(subset='patient_nbr', keep='first'))
print(f"\nRemoved {before - len(df_clean):,} duplicate patient encounters")

# patients who died or went to hospice can't be readmitted
# including them would pull the readmission rate down artificially
hospice = [11, 13, 14, 19, 20, 21]
before = len(df_clean)
df_clean = df_clean[~df_clean['discharge_disposition_id'].isin(hospice)]
print(f"Removed {before - len(df_clean):,} hospice/deceased records")
print(f"Clean dataset: {len(df_clean):,} rows")


# ── 4. Feature engineering ────────────────────────────────────────────────────

# binary target - 1 if readmitted within 30 days, 0 otherwise
df_clean['readmitted_binary'] = (df_clean['readmitted'] == '<30').astype(int)

# age comes as brackets like [50-60), converting to numeric midpoint
age_map = {
    '[0-10)': 5, '[10-20)': 15, '[20-30)': 25, '[30-40)': 35,
    '[40-50)': 45, '[50-60)': 55, '[60-70)': 65,
    '[70-80)': 75, '[80-90)': 85, '[90-100)': 95
}
df_clean['age_numeric'] = df_clean['age'].map(age_map)

def bucket_age(age_str):
    v = age_map.get(age_str, 0)
    if v < 40: return '<40'
    elif v < 60: return '40-59'
    elif v < 75: return '60-74'
    else: return '75+'

df_clean['age_group'] = df_clean['age'].apply(bucket_age)

# count how many of the 23 drugs were actually changed at discharge
# Up/Down = changed, No/Steady = not changed
med_cols = [
    'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
    'glimepiride', 'acetohexamide', 'glipizide', 'glyburide',
    'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',
    'miglitol', 'troglitazone', 'tolazamide', 'examide',
    'citoglipton', 'insulin', 'glyburide-metformin',
    'glipizide-metformin', 'glimepiride-pioglitazone',
    'metformin-rosiglitazone', 'metformin-pioglitazone'
]
for col in med_cols:
    df_clean[col + '_flag'] = df_clean[col].apply(
        lambda x: 0 if x in ['No', 'Steady'] else 1
    )
df_clean['total_medications_changed'] = df_clean[
    [c + '_flag' for c in med_cols]
].sum(axis=1)

df_clean['on_insulin'] = (df_clean['insulin'] != 'No').astype(int)

# patients with 3+ prior inpatient visits = high utilisers
# this turned out to be the strongest predictor in the analysis
df_clean['high_utiliser'] = (df_clean['number_inpatient'] >= 3).astype(int)

# length of stay bucket
df_clean['los_bucket'] = pd.cut(
    df_clean['time_in_hospital'],
    bins=[0, 3, 7, 14],
    labels=['Short (1-3d)', 'Medium (4-7d)', 'Long (8-14d)']
)

print(f"\nOverall 30-day readmission rate: {df_clean['readmitted_binary'].mean():.1%}")


# ── 5. Exploratory charts ─────────────────────────────────────────────────────

COLORS = ['#185FA5', '#E24B4A', '#1D9E75', '#EF9F27', '#7F77DD']

def save(fname):
    plt.tight_layout()
    plt.savefig(f'outputs/{fname}', dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()


# chart 1 - overall readmission breakdown
fig, ax = plt.subplots(figsize=(7, 4))
vals = [
    (df_clean['readmitted'] == 'NO').sum(),
    (df_clean['readmitted'] == '>30').sum(),
    (df_clean['readmitted'] == '<30').sum(),
]
labels = ['No readmission', 'Readmitted >30d', 'Readmitted <30d']
bars = ax.barh(labels, vals, color=['#1D9E75', '#EF9F27', '#E24B4A'], height=0.5)
for bar, v in zip(bars, vals):
    pct = v / sum(vals) * 100
    ax.text(bar.get_width() + 100, bar.get_y() + bar.get_height() / 2,
            f'{v:,}  ({pct:.1f}%)', va='center', fontsize=10)
ax.set_xlabel('Number of encounters')
ax.set_title('Readmission distribution', fontweight='bold', pad=10)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
save('01_readmission_distribution.png')


# chart 2 - by age group
age_order = ['<40', '40-59', '60-74', '75+']
age_rates = (df_clean.groupby('age_group')['readmitted_binary']
             .mean() * 100).reindex(age_order)

fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(age_rates.index, age_rates.values, color=COLORS[0], width=0.5)
for bar, v in zip(bars, age_rates.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
            f'{v:.1f}%', ha='center', va='bottom', fontsize=10)
avg = df_clean['readmitted_binary'].mean() * 100
ax.axhline(avg, color=COLORS[1], linestyle='--', linewidth=1.2,
           label=f'Overall avg ({avg:.1f}%)')
ax.set_ylabel('30-day readmission rate (%)')
ax.set_xlabel('Age group')
ax.set_title('Readmission rate by age group', fontweight='bold', pad=10)
ax.legend()
save('02_readmission_by_age.png')


# chart 3 - prior visits vs readmission rate (this was the most interesting finding)
visit_rates = (df_clean.groupby('number_inpatient')['readmitted_binary']
               .mean() * 100).head(10)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(visit_rates.index, visit_rates.values,
        marker='o', color=COLORS[0], linewidth=2, markersize=6)
ax.fill_between(visit_rates.index, visit_rates.values, alpha=0.1, color=COLORS[0])
ax.set_xlabel('Prior inpatient visits')
ax.set_ylabel('30-day readmission rate (%)')
ax.set_title('More prior visits → much higher readmission risk', fontweight='bold', pad=10)
save('03_readmission_by_prior_visits.png')

print(f"\n0 prior visits  → {visit_rates.get(0, 0):.1f}% readmission")
print(f"3+ prior visits → {visit_rates.get(3, 0):.1f}% readmission")


# chart 4 - length of stay
los_rates = (df_clean.groupby('los_bucket', observed=True)['readmitted_binary']
             .mean() * 100)

fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(los_rates.index.astype(str), los_rates.values,
              color=[COLORS[2], COLORS[3], COLORS[1]], width=0.5)
for bar, v in zip(bars, los_rates.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
            f'{v:.1f}%', ha='center', va='bottom', fontsize=10)
ax.set_ylabel('30-day readmission rate (%)')
ax.set_xlabel('Length of stay')
ax.set_title('Readmission rate by length of stay', fontweight='bold', pad=10)
save('04_readmission_by_los.png')


# chart 5 - top specialties (filtering to min 200 patients for reliability)
spec = (df_clean[df_clean['medical_specialty'] != 'Unknown']
        .groupby('medical_specialty')
        .agg(total=('readmitted_binary', 'count'),
             rate=('readmitted_binary', 'mean'))
        .query('total >= 200')
        .sort_values('rate', ascending=False)
        .head(10))
spec['rate_pct'] = spec['rate'] * 100

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.barh(spec.index[::-1], spec['rate_pct'][::-1], color=COLORS[0], height=0.6)
for bar, (_, row) in zip(bars, spec[::-1].iterrows()):
    ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
            f"{row['rate_pct']:.1f}%  (n={int(row['total']):,})",
            va='center', fontsize=9)
ax.set_xlabel('30-day readmission rate (%)')
ax.set_title('Top 10 specialties by readmission rate (min. 200 patients)',
             fontweight='bold', pad=10)
save('05_readmission_by_specialty.png')


# chart 6 - insulin change impact at discharge
med_impact = (df_clean.groupby('insulin')['readmitted_binary']
              .mean() * 100).reindex(['No', 'Down', 'Steady', 'Up'])

fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(med_impact.index, med_impact.values,
              color=[COLORS[2], COLORS[1], COLORS[3], COLORS[0]], width=0.5)
for bar, v in zip(bars, med_impact.values):
    if not np.isnan(v):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f'{v:.1f}%', ha='center', va='bottom', fontsize=10)
ax.set_ylabel('30-day readmission rate (%)')
ax.set_xlabel('Insulin status at discharge')
ax.set_title('Readmission rate by insulin change at discharge', fontweight='bold', pad=10)
save('06_insulin_impact.png')


# chart 7 - high utilisers vs everyone else
util_rates = (df_clean.groupby('high_utiliser')['readmitted_binary'].mean() * 100)

fig, ax = plt.subplots(figsize=(5, 4))
labels = ['Standard\n(< 3 prior visits)', 'High utilisers\n(3+ prior visits)']
bars = ax.bar(labels, util_rates.values,
              color=[COLORS[2], COLORS[1]], width=0.45)
for bar, v in zip(bars, util_rates.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
            f'{v:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
ax.set_ylabel('30-day readmission rate (%)')
ax.set_title('High utilisers have significantly higher readmission risk',
             fontweight='bold', pad=10)
save('07_high_utiliser_risk.png')


# chart 8 - correlation heatmap to see which features relate to readmission
num_cols = [
    'time_in_hospital', 'num_lab_procedures', 'num_procedures',
    'num_medications', 'number_outpatient', 'number_emergency',
    'number_inpatient', 'number_diagnoses', 'age_numeric',
    'total_medications_changed', 'readmitted_binary'
]
corr = df_clean[num_cols].corr()

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, linewidths=0.5, ax=ax,
            cbar_kws={'shrink': 0.8}, annot_kws={'size': 9})
ax.set_title('Correlation matrix — numeric features', fontweight='bold', pad=10)
save('08_correlation_heatmap.png')


# ── 6. Quick findings summary ─────────────────────────────────────────────────

overall = df_clean['readmitted_binary'].mean() * 100
hu_rate = df_clean[df_clean['high_utiliser'] == 1]['readmitted_binary'].mean() * 100

print(f"\n{'='*50}")
print("FINDINGS")
print(f"{'='*50}")
print(f"Overall readmission rate : {overall:.1f}%")
print(f"High utiliser rate       : {hu_rate:.1f}%  ({hu_rate/overall:.1f}x the average)")
print(f"Highest risk age group   : {age_rates.idxmax()}  ({age_rates.max():.1f}%)")


# ── 7. Save cleaned dataset ───────────────────────────────────────────────────

cols_to_keep = [
    'encounter_id', 'patient_nbr', 'race', 'gender', 'age', 'age_group',
    'age_numeric', 'admission_type_id', 'discharge_disposition_id',
    'admission_source_id', 'time_in_hospital', 'medical_specialty',
    'num_lab_procedures', 'num_procedures', 'num_medications',
    'number_outpatient', 'number_emergency', 'number_inpatient',
    'diag_1', 'diag_2', 'diag_3', 'number_diagnoses',
    'insulin', 'diabetesMed', 'readmitted', 'readmitted_binary',
    'high_utiliser', 'on_insulin', 'los_bucket', 'total_medications_changed'
]

df_final = df_clean[cols_to_keep]
df_final.to_csv('outputs/cleaned_readmission_data.csv', index=False)
print(f"\nSaved cleaned data → outputs/cleaned_readmission_data.csv")
print(f"Shape: {df_final.shape[0]:,} rows × {df_final.shape[1]} columns")
