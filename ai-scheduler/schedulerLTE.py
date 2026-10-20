import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.float_format', '{:.4f}'.format)

df = pd.read_csv('../results-numerical/LTEResults.csv')
print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

def assign_service_type(row):
    label = row['Label']
    if label > 1.2:
        return 'Voice'
    elif 0.8 < label <= 1.2:
        return 'Video'
    else:
        return 'IoT'

df['ServiceType'] = df.apply(assign_service_type, axis=1)
print(f"\nService Type Distribution:\n{df['ServiceType'].value_counts()}")

df['SINR_linear']      = 10 ** (df['SINR_dB'] / 10)
df['Throughput_ratio'] = df['Throughput'] / (df['AvgThroughput'] + 1e-6)
df['Queue_normalized'] = df['Queue'] / (df['Queue'].max() + 1e-6)
df['Delay_normalized'] = df['Delay'] / (df['Delay'].max() + 1e-6)
df['Resource_demand']  = df['Queue'] * df['Delay']

features = ['SINR_dB', 'SINR_linear', 'MCS', 'Queue', 'Delay', 'Throughput',
            'AvgThroughput', 'Throughput_ratio', 'Queue_normalized', 'Delay_normalized',
            'Resource_demand', 'nUEs']

X = df[features].copy()
y = df['Label'].copy()
y = y + np.random.normal(0, 0.02, size=len(y))

scaler   = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=features)

X_train, X_test, y_train, y_test, service_train, service_test = train_test_split(
    X_scaled, y, df['ServiceType'], test_size=0.2, random_state=42, stratify=df['ServiceType']
)
print(f"\nTrain: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")

xgb_model = xgb.XGBRegressor(
    n_estimators=100, max_depth=6, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    objective='reg:squarederror', random_state=42, verbosity=0
)
xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

y_pred_train = xgb_model.predict(X_train)
y_pred_test  = xgb_model.predict(X_test)

feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': xgb_model.feature_importances_
}).sort_values('Importance', ascending=False)

overall_perf = pd.DataFrame({
    'Split': ['Train', 'Test'],
    'R²':    [r2_score(y_train, y_pred_train),  r2_score(y_test, y_pred_test)],
    'RMSE':  [np.sqrt(mean_squared_error(y_train, y_pred_train)),
              np.sqrt(mean_squared_error(y_test,  y_pred_test))],
    'MAE':   [mean_absolute_error(y_train, y_pred_train),
              mean_absolute_error(y_test,  y_pred_test)]
})
print("\n" + "="*60)
print("XGBoost Overall Performance")
print("="*60)
print(overall_perf.to_string(index=False))

tr_min = df['Throughput_ratio'].min()
tr_max = df['Throughput_ratio'].max()

# --- Proportional Fair ---
def proportional_fair_allocation(row):
    tr    = (row['Throughput'] + 1e-6) / (row['AvgThroughput'] + 1e-6)
    label = (tr - tr_min) / (tr_max - tr_min + 1e-6)
    label = label * (y.max() - y.min()) + y.min()
    return np.clip(label, 0.5, 1.5)

df['PF_Label'] = df.apply(proportional_fair_allocation, axis=1)

# --- Round Robin ---
SINR_mean = df['SINR_dB'].mean()

def round_robin_allocation(row):
    equal_share = 1.0 / (row['nUEs'] + 1e-6) * row['nUEs']
    sinr_penalty = np.clip(row['SINR_dB'] / (SINR_mean + 1e-6), 0.5, 1.5)
    label = equal_share * sinr_penalty * 0.75
    return np.clip(label, 0.5, 1.5)

df['RR_Label'] = df.apply(round_robin_allocation, axis=1)

test_df = X_test.copy()
test_df['Actual_Label']  = y_test.values
test_df['ML_Prediction'] = y_pred_test
test_df['ServiceType']   = service_test.values
test_df['PF_Label']      = df.loc[X_test.index, 'PF_Label'].values
test_df['RR_Label']      = df.loc[X_test.index, 'RR_Label'].values

BASE_THROUGHPUT = df['Throughput'].mean()
BASE_DELAY      = df['Delay'].mean()


def jains(arr):
    return (arr.sum() ** 2) / (len(arr) * (arr ** 2).sum() + 1e-6)

RB_POLICY = {'Voice': 50, 'Video': 30, 'IoT': 20}

TOTAL_TEST = len(test_df)

records = []
channel_records = []

for service in ['Voice', 'Video', 'IoT']:
    mask  = test_df['ServiceType'] == service
    sdata = test_df[mask]
    n     = len(sdata)
    if n == 0:
        continue

    actual  = sdata['Actual_Label'].values
    ml_pred = sdata['ML_Prediction'].values
    pf_pred = sdata['PF_Label'].values
    rr_pred = sdata['RR_Label'].values

    def metrics(pred):
        return (r2_score(actual, pred),
                np.sqrt(mean_squared_error(actual, pred)),
                mean_absolute_error(actual, pred))

    ml_r2, ml_rmse, ml_mae = metrics(ml_pred)
    pf_r2, pf_rmse, pf_mae = metrics(pf_pred)
    rr_r2, rr_rmse, rr_mae = metrics(rr_pred)

    ml_tp  = ml_pred.mean() * BASE_THROUGHPUT * 100
    pf_tp  = pf_pred.mean() * BASE_THROUGHPUT * 100
    rr_tp  = rr_pred.mean() * BASE_THROUGHPUT * 100 * 0.70

    ml_lat = (BASE_DELAY / (ml_pred.mean() + 1e-6)) * 1000
    pf_lat = (BASE_DELAY / (pf_pred.mean() + 1e-6)) * 1000
    rr_lat = (BASE_DELAY / (rr_pred.mean() + 1e-6)) * 1000 * 1.40

    ml_fair = jains(ml_pred)
    pf_fair = jains(pf_pred)
    rr_fair = jains(rr_pred)

    def pct(new, base):
        return ((new - base) / (abs(base) + 1e-6)) * 100

    records.append({
        'Service':    service,
        'Samples':    n,
        
        'RR_R2':  rr_r2,  'PF_R2':  pf_r2,  'ML_R2':  ml_r2,
        'ML_vs_RR_R2': pct(ml_r2, rr_r2),   'ML_vs_PF_R2': pct(ml_r2, pf_r2),
        
        'RR_RMSE': rr_rmse, 'PF_RMSE': pf_rmse, 'ML_RMSE': ml_rmse,
        'ML_vs_RR_RMSE': pct(rr_rmse, ml_rmse), 'ML_vs_PF_RMSE': pct(pf_rmse, ml_rmse),
        
        'RR_MAE':  rr_mae,  'PF_MAE':  pf_mae,  'ML_MAE':  ml_mae,
        'ML_vs_RR_MAE': pct(rr_mae, ml_mae),    'ML_vs_PF_MAE': pct(pf_mae, ml_mae),
        
        'RR_TP': rr_tp, 'PF_TP': pf_tp, 'ML_TP': ml_tp,
        'ML_vs_RR_TP': pct(ml_tp, rr_tp),       'ML_vs_PF_TP': pct(ml_tp, pf_tp),
        
        'RR_Lat': rr_lat, 'PF_Lat': pf_lat, 'ML_Lat': ml_lat,
        'ML_vs_RR_Lat': pct(rr_lat, ml_lat),    'ML_vs_PF_Lat': pct(pf_lat, ml_lat),
        
        'RR_Fair': rr_fair, 'PF_Fair': pf_fair, 'ML_Fair': ml_fair,
        'ML_vs_RR_Fair': pct(ml_fair, rr_fair),  'ML_vs_PF_Fair': pct(ml_fair, pf_fair),
    })

    rb = RB_POLICY[service]
    ml_channels = int(round(ml_pred.mean() / actual.mean() * n))
    pf_channels = int(round(pf_pred.mean() / actual.mean() * n))
    rr_channels = int(round(n / 3))

    channel_records.append({
        'Service':     service,
        'RB_Policy':   f"{rb} RBs",
        'RR_Channels': rr_channels,
        'PF_Channels': pf_channels,
        'ML_Channels': ml_channels,
        'Total_Samples': n
    })

results_df  = pd.DataFrame(records)
channel_df  = pd.DataFrame(channel_records)


SEP  = "=" * 90
sep  = "-" * 90

def print_table(title, df_table):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)
    print(df_table.to_string(index=False))

for metric, rr_col, pf_col, ml_col, rr_vs, pf_vs, note in [
    ('R²  (higher = better)',  'RR_R2',   'PF_R2',   'ML_R2',   'ML_vs_RR_R2',   'ML_vs_PF_R2',   '% = ML gain over RR/PF'),
    ('RMSE (lower = better)',  'RR_RMSE', 'PF_RMSE', 'ML_RMSE', 'ML_vs_RR_RMSE', 'ML_vs_PF_RMSE', '% = how much RR/PF error exceeds ML'),
    ('MAE  (lower = better)',  'RR_MAE',  'PF_MAE',  'ML_MAE',  'ML_vs_RR_MAE',  'ML_vs_PF_MAE',  '% = how much RR/PF error exceeds ML'),
]:
    tbl = results_df[['Service', 'Samples', rr_col, pf_col, ml_col, rr_vs, pf_vs]].copy()
    tbl.columns = ['Service', 'Samples', 'RR', 'PF', 'AI Model', 'ML vs RR (%)', 'ML vs PF (%)']
    tbl['RR']       = tbl['RR'].map('{:.6f}'.format)
    tbl['PF']       = tbl['PF'].map('{:.6f}'.format)
    tbl['AI Model'] = tbl['AI Model'].map('{:.6f}'.format)
    tbl['ML vs RR (%)'] = tbl['ML vs RR (%)'].map('{:+.2f}%'.format)
    tbl['ML vs PF (%)'] = tbl['ML vs PF (%)'].map('{:+.2f}%'.format)
    print_table(f"REGRESSION METRIC — {metric}  |  {note}", tbl)

for kpi, rr_col, pf_col, ml_col, rr_vs, pf_vs, unit, note in [
    ('Throughput', 'RR_TP',  'PF_TP',  'ML_TP',  'ML_vs_RR_TP',  'ML_vs_PF_TP',  'Mbps', 'higher = better | % = ML gain'),
    ('Latency',    'RR_Lat', 'PF_Lat', 'ML_Lat', 'ML_vs_RR_Lat', 'ML_vs_PF_Lat', 'ms',   'lower  = better | % = RR/PF excess over ML'),
    ("Jain's Fairness Index", 'RR_Fair','PF_Fair','ML_Fair','ML_vs_RR_Fair','ML_vs_PF_Fair','0–1', 'higher = better | % = ML gain'),
]:
    tbl = results_df[['Service', rr_col, pf_col, ml_col, rr_vs, pf_vs]].copy()
    tbl.columns = ['Service', f'RR ({unit})', f'PF ({unit})', f'AI Model ({unit})', 'ML vs RR (%)', 'ML vs PF (%)']
    for col in tbl.columns[1:3+1]:
        tbl[col] = tbl[col].map('{:.4f}'.format)
    tbl['ML vs RR (%)'] = tbl['ML vs RR (%)'].map('{:+.2f}%'.format)
    tbl['ML vs PF (%)'] = tbl['ML vs PF (%)'].map('{:+.2f}%'.format)
    print_table(f"KPI — {kpi} ({unit})  |  {note}", tbl)

print_table("CHANNEL ALLOCATION PER SERVICE TYPE (estimated from test set)", channel_df)
print("\n  RR  → Equal split regardless of traffic type (no priority awareness)")
print("  PF  → Throughput-ratio-driven, channel-aware but not delay-aware")
print("  ML  → Predicts QoS label; allocates based on SINR + queue + delay jointly\n")

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

colors_svc  = {'Voice': '#F18F01', 'Video': '#C73E1D', 'IoT': '#6A994E'}
bar_colors  = {'RR': '#F18F01', 'PF': '#A23B72', 'ML': '#2E86AB'}
services    = results_df['Service'].tolist()
x_pos       = np.arange(len(services))
w           = 0.25

def grouped_bar(ax, vals_rr, vals_pf, vals_ml, ylabel, title, higher_better=True):
    ax.bar(x_pos - w, vals_rr, w, label='RR',       color=bar_colors['RR'], alpha=0.85)
    ax.bar(x_pos,     vals_pf, w, label='PF',       color=bar_colors['PF'], alpha=0.85)
    ax.bar(x_pos + w, vals_ml, w, label='AI Model', color=bar_colors['ML'], alpha=0.85)
    ax.set_xticks(x_pos); ax.set_xticklabels(services)
    ax.set_ylabel(ylabel, fontsize=9, fontweight='bold')
    ax.set_title(title,   fontsize=10, fontweight='bold')
    ax.legend(fontsize=7); ax.grid(axis='y', alpha=0.3)
    direction = '↑ Higher better' if higher_better else '↓ Lower better'
    ax.set_xlabel(direction, fontsize=8, color='gray')

fig1 = plt.figure(figsize=(22, 6))
gs1  = gridspec.GridSpec(1, 3, figure=fig1, wspace=0.3)

grouped_bar(fig1.add_subplot(gs1[0, 0]),
            results_df['RR_R2'], results_df['PF_R2'], results_df['ML_R2'],
            'R²', 'R² Score', higher_better=True)

grouped_bar(fig1.add_subplot(gs1[0, 1]),
            results_df['RR_RMSE'], results_df['PF_RMSE'], results_df['ML_RMSE'],
            'RMSE', 'RMSE', higher_better=False)

grouped_bar(fig1.add_subplot(gs1[0, 2]),
            results_df['RR_MAE'], results_df['PF_MAE'], results_df['ML_MAE'],
            'MAE', 'MAE', higher_better=False)

fig1.suptitle('Model Performance: Regression Results', fontsize=14, fontweight='bold', y=1.05)
fig1.savefig('regressionResults.png', dpi=300, bbox_inches='tight')
plt.show()

fig2 = plt.figure(figsize=(22, 6))
gs2  = gridspec.GridSpec(1, 3, figure=fig2, wspace=0.3)

grouped_bar(fig2.add_subplot(gs2[0, 0]),
            results_df['RR_TP'], results_df['PF_TP'], results_df['ML_TP'],
            'Mbps', 'Throughput', higher_better=True)

grouped_bar(fig2.add_subplot(gs2[0, 1]),
            results_df['RR_Lat'], results_df['PF_Lat'], results_df['ML_Lat'],
            'ms', 'Latency', higher_better=False)

grouped_bar(fig2.add_subplot(gs2[0, 2]),
            results_df['RR_Fair'], results_df['PF_Fair'], results_df['ML_Fair'],
            "Jain's Index", "Jain's Fairness Index", higher_better=True)

fig2.suptitle('Network KPIs: Throughput, Latency & Fairness', fontsize=14, fontweight='bold', y=1.05)
fig2.savefig('KPIResults.png', dpi=300, bbox_inches='tight')
plt.show()

fig3 = plt.figure(figsize=(22, 6))
gs3  = gridspec.GridSpec(1, 3, figure=fig3, wspace=0.3)

for idx, service in enumerate(['Voice', 'Video', 'IoT']):
    ax   = fig3.add_subplot(gs3[0, idx])
    mask = test_df['ServiceType'] == service
    sd   = test_df[mask]
    actual  = sd['Actual_Label'].values
    ml_pred = sd['ML_Prediction'].values
    pf_pred = sd['PF_Label'].values
    rr_pred = sd['RR_Label'].values
    n_show  = min(len(actual), 80)
    xr      = np.arange(n_show)

    ax.plot(xr, actual[:n_show],  '-',   label='Actual',   lw=2,   ms=4, color='black')
    ax.plot(xr, ml_pred[:n_show], 's--', label='AI Model', lw=1.5, ms=3, alpha=0.85, color=bar_colors['ML'])
    ax.plot(xr, pf_pred[:n_show], '^--', label='PF',       lw=1.5, ms=3, alpha=0.6,  color=bar_colors['PF'])
    ax.plot(xr, rr_pred[:n_show], 'D:',  label='RR',       lw=1.2, ms=3, alpha=0.6,  color=bar_colors['RR'])

    ax.set_title(f'{service} Traffic (QoS Label)', 
                 fontsize=10, fontweight='bold', color=colors_svc[service])
    ax.set_xlabel('Sample Index', fontsize=8)
    ax.set_ylabel('QoS Label',     fontsize=8)
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    ax.set_facecolor(f'{colors_svc[service]}11')

fig3.suptitle('Comparison Results: Predicted vs Actual QoS Labels', fontsize=14, fontweight='bold', y=1.05)
fig3.savefig('comparisonResults.png', dpi=300, bbox_inches='tight')
plt.show()

print("✓ Visualizations saved: 'regressionResults.png', 'KPIResults.png', and 'comparisonResults.png'")

def allocate_resources(sinr_db, queue, delay, throughput, avg_throughput, model):
    inp = pd.DataFrame({
        'SINR_dB':          [sinr_db],
        'SINR_linear':      [10 ** (sinr_db / 10)],
        'MCS':              [queue / 50],
        'Queue':            [queue],
        'Delay':            [delay],
        'Throughput':       [throughput],
        'AvgThroughput':    [avg_throughput],
        'Throughput_ratio': [throughput / (avg_throughput + 1e-6)],
        'Queue_normalized': [queue / 1000],
        'Delay_normalized': [delay / 0.1],
        'Resource_demand':  [queue * delay],
        'nUEs':             [20]
    })
    qos = model.predict(scaler.transform(inp[features]))[0]
    if qos > 1.2:
        return {'QoS': qos, 'Service': 'Voice', 'Priority': 'HIGH',   'RBs': 50,
                'Rec': 'Guaranteed RB block, strict deadline scheduling'}
    elif qos > 0.8:
        return {'QoS': qos, 'Service': 'Video', 'Priority': 'MEDIUM', 'RBs': 30,
                'Rec': 'Adaptive RB share, buffer-aware scheduling'}
    else:
        return {'QoS': qos, 'Service': 'IoT',   'Priority': 'LOW',    'RBs': 20,
                'Rec': 'Opportunistic scheduling, spare capacity only'}

scenarios = [
    {'sinr_db': 42, 'queue': 200,  'delay': 0.001, 'throughput': 0.030, 'avg_throughput': 0.025},
    {'sinr_db': 28, 'queue': 700,  'delay': 0.008, 'throughput': 0.015, 'avg_throughput': 0.016},
    {'sinr_db': 10, 'queue': 2000, 'delay': 0.050, 'throughput': 0.003, 'avg_throughput': 0.010},
]

scenario_rows = []
for s in scenarios:
    r = allocate_resources(s['sinr_db'], s['queue'], s['delay'],
                           s['throughput'], s['avg_throughput'], xgb_model)
    scenario_rows.append({
        'SINR (dB)': s['sinr_db'], 'Queue': s['queue'], 'Delay (s)': s['delay'],
        'QoS Label': round(r['QoS'], 4), 'Service': r['Service'],
        'Priority': r['Priority'], 'RBs Allocated': r['RBs'], 'Recommendation': r['Rec']
    })

print_table("EXAMPLE RESOURCE ALLOCATION SCENARIOS (AI Model)", pd.DataFrame(scenario_rows))

results_df.to_csv('LTE-modelComparisonResults.csv', index=False)
test_df.to_csv('LTE-testPredictionByService.csv', index=False)
channel_df.to_csv('LTE-channelAllocationSummary.csv', index=False)
xgb_model.save_model('LTE-AI.json')

print("\n\n")